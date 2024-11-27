> 回归 老本行

偶然得到 nailong 数据集, 分为两块, 一种是给[分类模型使用的数据集](https://huggingface.co/datasets/refoundd/NailongClassification), 另一种是给[目标检测模型使用的数据集](https://huggingface.co/datasets/refoundd/NailongDetection)

> 后者的数据量不是非常多(6张), 等到有足够多的数据或者我一时兴起手动标注在进行研究

## 初版方案

### 数据集选择

在我刚接触这个数据集的时候, 数据集是只有奶龙的(无其他标签的数据), 这个时候第一个想法就是引入其他分类, 这里采用cifer10数据集的数据, 对数据集进行增广

然而, cifer10 的数据分布毕竟与常见群聊内发送的图片不同, 我觉得会影响最终能力, 应该有选择性而不是随意添加其他类型的图片, 在一番搜索之后, 选中了 [表情包数据集](https://github.com/LLM-Red-Team/emo-visual-data)

虽然这个数据集的原计划是用来检测 VLLM 的能力, 但我认为在我们这个任务中也可以使用

### 模型

在敲定数据集之后, 就开始挑选模型了, 因为是个人小项目, 这里采用我个人喜好的模型选择, 使用了 [convnext 系模型](https://github.com/facebookresearch/ConvNeXt)

这个模型的论文是一篇非常经典的实验文, 里面大量探索了一些技巧对模型能力的影响 (各类消融实验), 虽然他是 2020 年推出, 但他对现在的卷积网络的训练技巧的指引很大

具体细节可以搜索相关的模型解析, 这里不再赘述

<details><summary>model.py</summary>
<p>

```python
# copy from facebook/ConvNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
```

</p>
</details> 


### 代码

训练代码大部分都是模板, 不过, 我发现我还没有一个属于自己的 trainer, 趁着这次训练模型的时候补充一个

使用别人的 trainer 难免会遇到 debug, 而代码不熟的情况, 自己写的 trainer 可以掌握各种细节

在整理了一些以前代码后, 总结出了覆盖许多训练模型情况的流程, 趁着这个时候测试一下现在ai编码的能力, 将流程发给 claude-sonnet 后, 输出了一版代码, 在我的一些小修小补(补充日志)后, 就可以跑起来了

<details><summary>trainer.py</summary>
<p>

```python
import gc
import json
import logging
import os
import shutil

import torch
from torch import optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

example_config = {
    "model": "model_name",
    "checkpoint_dir": "./checkpoints",
    "tensorboard_dir": "./tensorboard",
    "device": "cuda",
    "enable_cudnn_benchmark": True,
    "enable_amp": False,
    "learning_rate": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "enable_compile": False,
    "weight_decay": 0.05,
    "max_steps": 100000,
    "max_grad_norm": 1.0,
    "save_every": 10000,
    "gradient_accumulation_steps": 4
}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_training()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.config['tensorboard_dir'])
        
    def setup_device(self):
        """设置设备"""
        self.device = torch.device(self.config['device'])
        torch.backends.cudnn.benchmark = self.config.get('enable_cudnn_benchmark', True)
        if self.device.type == 'cuda':
            self.logger.info(f'Using device: {self.device} ({torch.cuda.get_device_name()})')
        else:
            self.logger.info(f'Using device: {self.device}')
            
    def setup_model(self):
        """设置模型、损失函数等"""
        self.model = self.build_model().to(self.device)
        if self.config.get('enable_compile', False):
            self.model.compile()
        self.criterion = self.build_criterion()
        
        # 打印模型信息
        n_parameters = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f'Number of parameters: {n_parameters:,}')
        
    def setup_training(self):
        """设置训练相关组件"""
        # 优化器
        self.optimizer = self.build_optimizer()
        
        # 学习率调度器
        self.scheduler = self.build_scheduler()
        
        # 梯度缩放器(用于混合精度训练)
        self.scaler = GradScaler(
            enabled=self.config.get('enable_amp', False)
        )
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # 加载检查点
        self.steps = 0
        self.best_metric = {}
        self.load_checkpoint()
        
    def build_model(self):
        """构建模型(需要子类实现)"""
        raise NotImplementedError
        
    def build_criterion(self):
        """构建损失函数(需要子类实现)"""
        raise NotImplementedError
        
    def build_optimizer(self):
        """构建优化器"""
        # 区分需要和不需要weight decay的参数
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        opt_params = [
            {'params': decay_params, 'weight_decay': self.config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(
            opt_params,
            lr=self.config['learning_rate'],
            betas=self.config.get('betas', (0.9, 0.999)),
            eps=self.config.get('eps', 1e-8)
        )
        
    def build_scheduler(self):
        """构建学习率调度器(需要子类实现)"""
        return NotImplementedError
        
    def build_dataloader(self):
        """构建数据加载器(需要子类实现)"""
        raise NotImplementedError
        
    def train_step(self, batch):
        """单步训练(需要子类实现)"""
        raise NotImplementedError
        
    def validate(self):
        """验证(需要子类实现)"""
        raise NotImplementedError
        
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'steps': self.steps,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(
            state,
            os.path.join(self.config['checkpoint_dir'], 'latest.pt')
        )
        
        # 保存最佳检查点
        if is_best:
            shutil.copy(
                os.path.join(self.config['checkpoint_dir'], 'latest.pt'),
                os.path.join(self.config['checkpoint_dir'], 'best.pt')
            )
            
    def load_checkpoint(self):
        """加载检查点"""
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            'latest.pt'
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )
            
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.steps = checkpoint['steps']
            self.best_metric = checkpoint['best_metric']
            
            self.logger.info(f'Loaded checkpoint from {checkpoint_path}')
            self.logger.info(f'Training will resume from step {self.steps}')
    
    @staticmethod
    def is_better_performance(baseline_dict, compare_dict):
        """
        判断compare_dict中的指标是否全面超过baseline_dict
        
        Args:
            baseline_dict: 基准字典,格式为 {指标名: 值}
            compare_dict: 比较字典,格式为 {指标名: 值} 
        
        Returns:
            bool: 如果compare_dict中所有指标都严格大于baseline_dict则返回True,否则返回False
        """
        if not baseline_dict:
            return True
        
        # 检查两个字典的键是否一致
        if set(baseline_dict.keys()) != set(compare_dict.keys()):
            return False
            
        # 检查每个指标是否都有提升
        for metric in baseline_dict:
            if compare_dict[metric] <= baseline_dict[metric]:
                return False
                
        return True
            
    def train(self):
        """训练流程"""
        train_loader = self.build_dataloader()
        self.model.train()
        
        self.logger.info('Start training...')
        pbar = tqdm(total=self.config['max_steps'], initial=self.steps)
        
        while self.steps < self.config['max_steps']:
            for batch in train_loader:
                # 训练一步
                with torch.autocast(device_type=self.config['device'], enabled=self.config.get('enable_amp', False)):
                    loss = self.train_step(batch)
                self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
                
                if (self.steps + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.config.get('max_grad_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )

                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                # 记录
                self.writer.add_scalar('train/loss', loss, self.steps)
                self.writer.add_scalar(
                    'train/lr',
                    self.scheduler.get_last_lr()[0],
                    self.steps
                )
                
                self.steps += 1
                pbar.update(1)
                
                # 验证和保存
                if self.steps % self.config['save_every'] == 0:
                    metric = self.validate()
                    for i in metric:
                        self.logger.info(f'Validation {i}: {metric[i]}')
                        self.writer.add_scalar(f'val/{i}', metric[i], self.steps)
                    
                    is_best = self.is_better_performance(self.best_metric, metric)
                    if is_best:
                        self.best_metric = metric

                    self.model.train()
                    self.save_checkpoint(is_best)
                    
                if self.steps >= self.config['max_steps']:
                    break
                
            gc.collect()
            torch.cuda.empty_cache()
                    
        pbar.close()
        self.logger.info('Training finished!')


def main():
    """主函数"""
    # 加载配置
    with open('config.json') as f:
        config = json.load(f)
        
    # 创建输出目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['tensorboard_dir'], exist_ok=True)
    
    # 训练
    trainer = Trainer(config)
    trainer.train()
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
``` 

</p>
</details> 

<details><summary>train.py(代码未整理完毕)</summary>
<p>

```python

```

</p>
</details> 

> TODO 补充trainer的用例与用法
> TODO 补充超参搜索相关内容, 以及其用例与用法

### 数据增广

原始数据集只有两百多张图片, 这个时候无法避免的要做数据增广, 扩展 nailong 标签的数据, 这里因为是初版方案, 也没有非常精细的增广方案, 这里使用了以下几种方式(代码在如上train.py中):

- 给图片添加颜色遮罩
    让模型不要将遇到黄色的就判定为奶龙
- 在负样本中嵌入正样本
    很经典的增广数据的手法
- 图片轴对称
- 等
