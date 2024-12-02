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

`trainer.py` 简单整合了几个常用的训练手段, 比如混合精度训练, 梯度裁剪, 梯度累计, weight_decay(写死了), tensorboard的记录, 断点续训等操作, 需要注意的是, `trainer.py` 没有使用 epoch 作为训练进度, 而是用了更精细的 step (每次迭代参数即为一个step), 使用的时候需要自行实现模型构建, 损失函数构建 学习率调度器 数据集加载器, 单步训练, 验证的流程的子类实现

然后将一些配置放到config中便于读取, 其中有一些配置是必须的, 其他则是子类实现的时候需要的

听起来可能有点抽象, 下面是一个简单的trainer使用案例

<details><summary>trainer使用案例</summary>
<p>

import torchvision
import torch
from trainer import Trainer
from torchvision.models import resnet18
from torch.optim.lr_scheduler import LambdaLR



transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ConstantLambdaLR(LambdaLR):
    def __init__(self, optimizer, **kwargs):
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def _step_inner(self, steps):
        return 1


class Cifer10Trainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        model = resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        return model
    
    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()
    
    def build_scheduler(self):
        return ConstantLambdaLR(self.optimizer)
    
    def build_dataloader(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./temp', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=1)
        return train_loader
    
    def train_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss
    
    def validate(self):
        self.model.eval()
        test_dataset = torchvision.datasets.CIFAR10(root='./temp', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=1)
        acc = []
        with torch.inference_mode():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                y_hat = self.model(inputs)
                acc.append((y_hat.argmax(dim=1) == labels).sum().item() / labels.size(0))
                
        return {'acc': sum(acc) / len(acc)}
                

def main():
    config = {
        "model": "resnet18",
        "checkpoint_dir": "./checkpoints",
        "tensorboard_dir": "./tensorboard",
        "device": "cuda",
        "enable_cudnn_benchmark": True,
        "enable_amp": False,
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "enable_compile": False,
        "weight_decay": 0.05,
        "max_steps": 500,
        "max_grad_norm": 1.0,
        "save_every": 100,
        "gradient_accumulation_steps": 1,
        'batch_size': 32
    }
    trainer = Cifer10Trainer(config)
    trainer.train()
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

</p>
</details> 

> 代码使用cifer10数据集, resnet18作为模型训练的简单的流程

有了流程接下来编写我们的训练代码

<details><summary>train.py(代码未整理完毕, 非初版代码, 仅供参考)</summary>
<p>

```python
import os
import random

import cv2
import numpy as np
from sklearn.metrics import f1_score
import torch
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms

# from torchvision.models import resnet18
from model import convnext_base
from trainer import Trainer

image_size = 224
batch_size = 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def get_color_from_image(image_path):
#     """
#     从纯色图片中获取RGB颜色值
#     返回: (R, G, B)元组
#     """
#     # 读取图片
#     image = Image.open(image_path).convert('RGB')
#     # 转换为numpy数组
#     img_array = np.array(image)
    
#     # 获取图片中心点的颜色值
#     h, w = img_array.shape[:2]
#     center_color = img_array[h//2, w//2]
    
#     # 或者计算整个图片的平均颜色
#     average_color = img_array.mean(axis=(0,1)).astype(int)
    
#     return tuple(average_color)  # 或者 tuple(average_color)


# class AugmentationUtils:
#     @staticmethod
#     def add_color_mask(image, is_positive):
#         """给图片添加颜色遮罩"""
#         # 转换为numpy数组并确保类型为uint8
#         image = np.array(image, dtype=np.uint8)
        
#         # 创建与图像相同大小的遮罩
#         mask = np.ones_like(image, dtype=np.uint8)
        
#         # 随机生成颜色
#         if is_positive:
#             color = [random.randint(0, 255) for _ in range(3)]
#         else:
#             color = get_color_from_image('22.png')
        
#         # 为遮罩赋予颜色    
#         for i in range(3):
#             mask[:, :, i] = color[i]
        
#         # 确保mask也是uint8类型
#         mask = mask.astype(np.uint8)
        
#         # 添加遮罩
#         alpha = 0.5  # 透明度
#         image = cv2.addWeighted(image, 1-alpha, mask, alpha, 0)
        
#         return Image.fromarray(image)

#     @staticmethod
#     def embed_positive_in_negative(positive_img, negative_img):
#         """在负样本中嵌入正样本"""
#         # 转换为numpy数组
#         pos_img = np.array(positive_img)
#         neg_img = np.array(negative_img)
        
#         # 确保图像是3通道的
#         if len(pos_img.shape) == 2:
#             pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2BGR)
#         if len(neg_img.shape) == 2:
#             neg_img = cv2.cvtColor(neg_img, cv2.COLOR_GRAY2BGR)
        
#         # 获取负样本尺寸
#         h, w = neg_img.shape[:2]
#         pos_h, pos_w = pos_img.shape[:2]
        
#         # 计算合适的缩放比例
#         scale = min(
#             random.uniform(0.5, 0.8),
#             (w * 0.8) / pos_w,
#             (h * 0.8) / pos_h
#         )
        
#         # 缩放正样本
#         new_size = (int(pos_w * scale), int(pos_h * scale))
#         pos_img_resized = cv2.resize(pos_img, new_size)
        
#         # 确保有效的随机位置范围
#         max_x = max(0, w - new_size[0])
#         max_y = max(0, h - new_size[1])
        
#         # 随机选择插入位置
#         x = random.randint(0, max_x) if max_x > 0 else 0
#         y = random.randint(0, max_y) if max_y > 0 else 0
        
#         # 获取ROI区域并确保与缩放后的正样本具有相同的形状
#         roi = neg_img[y:y+new_size[1], x:x+new_size[0]]
        
#         # 确保ROI和pos_img_resized具有相同的形状和通道数
#         if roi.shape == pos_img_resized.shape:
#             # 混合图像
#             blended = cv2.addWeighted(roi, 0.3, pos_img_resized, 0.7, 0)
#             neg_img[y:y+new_size[1], x:x+new_size[0]] = blended
        
#         return Image.fromarray(neg_img)
    
#     @staticmethod
#     def embed_same(positive_img, negative_img):
#         """在负样本中嵌入正样本"""
#         # 转换为numpy数组
#         pos_img = np.array(positive_img)
#         neg_img = np.array(negative_img)
        
#         # 确保图像是3通道的
#         if len(pos_img.shape) == 2:
#             pos_img = cv2.cvtColor(pos_img, cv2.COLOR_GRAY2BGR)
#         if len(neg_img.shape) == 2:
#             neg_img = cv2.cvtColor(neg_img, cv2.COLOR_GRAY2BGR)
        
#         # 获取负样本尺寸
#         h, w = neg_img.shape[:2]
#         pos_h, pos_w = pos_img.shape[:2]
        
#         # 计算合适的缩放比例
#         scale = min(
#             random.uniform(0.5, 0.8),
#             (w * 0.8) / pos_w,
#             (h * 0.8) / pos_h
#         )
        
#         # 缩放正样本
#         new_size = (int(pos_w * scale), int(pos_h * scale))
#         pos_img_resized = cv2.resize(pos_img, new_size)
        
#         # 确保有效的随机位置范围
#         max_x = max(0, w - new_size[0])
#         max_y = max(0, h - new_size[1])
        
#         # 随机选择插入位置
#         x = random.randint(0, max_x) if max_x > 0 else 0
#         y = random.randint(0, max_y) if max_y > 0 else 0
        
#         # 获取ROI区域并确保与缩放后的正样本具有相同的形状
#         roi = neg_img[y:y+new_size[1], x:x+new_size[0]]
        
#         # 确保ROI和pos_img_resized具有相同的形状和通道数
#         if roi.shape == pos_img_resized.shape:
#             # 混合图像
#             blended = cv2.addWeighted(roi, 0.3, pos_img_resized, 0.7, 0)
#             neg_img[y:y+new_size[1], x:x+new_size[0]] = blended
        
#         return Image.fromarray(neg_img)

#     @staticmethod
#     def flip_image(image):
#         """图片轴对称"""
#         return Image.fromarray(np.array(image)[:, ::-1])
    
#     @staticmethod
#     def mirror_half_image(image):
#         img_array = np.array(image)
    
#         # 获取图片尺寸
#         h, w = img_array.shape[:2]
        
#         # 取左半边
#         half_w = w // 2
#         left_half = img_array[:, :half_w]
        
#         # 水平翻转左半边得到右半边
#         right_half = left_half[:, ::-1]
        
#         # 拼接两个半边
#         mirrored = np.concatenate([left_half, right_half], axis=1)
        
#         return Image.fromarray(mirrored)
    

# def augment_dataset(positive_images, negative_images):
#     aug_utils = AugmentationUtils()
#     augmented_data = []
    
#     # 增强正样本
#     for pos_img in positive_images:
#         img = Image.open(pos_img).convert('RGB')
#         # 原图
#         augmented_data.append((img, 1))
#         # 颜色遮罩
#         augmented_data.append((aug_utils.add_color_mask(img, True), 1))
#         # 轴对称
#         augmented_data.append((aug_utils.flip_image(img), 1))
#         # 镜像一半
#         augmented_data.append((aug_utils.mirror_half_image(img), 1))
#         # 嵌入相同
#         img_id = random.randint(0, len(positive_images)-1)
#         aaa = Image.open(positive_images[img_id]).convert('RGB')
#         augmented_data.append((aug_utils.embed_same(aaa, img), 1))
        
    
#     # 增强负样本
#     for i, neg_img in enumerate(negative_images):
#         img = Image.open(neg_img).convert('RGB')
#         # 原图
#         augmented_data.append((img, 0))
#         # 颜色遮罩
#         augmented_data.append((aug_utils.add_color_mask(img, False), 0))
#         # 镜像一半
#         augmented_data.append((aug_utils.mirror_half_image(img), 0))
#         # 嵌入正样本
#         pos_img = Image.open(positive_images[random.randint(0, len(positive_images)-1)]).convert('RGB')
#         augmented_data.append((aug_utils.embed_positive_in_negative(pos_img, img), 1))
#         # 嵌入相同
#         img_id = random.randint(0, len(negative_images)-1)
#         aaa = Image.open(negative_images[img_id]).convert('RGB')
#         augmented_data.append((aug_utils.embed_same(aaa, img), 0))
        

        
#     # # 显示并保存
#     # for i, (img, label) in enumerate(augmented_data):
#     #     # img.show()
#     #     os.makedirs('aug_images', exist_ok=True)
#     #     img.save(f'aug_images/aug_{i}.jpg')
    
#     # 统计
#     print(f"Positive: {len([x for x, y in augmented_data if y == 1])}, Negative: {len([x for x, y in augmented_data if y == 0])}")
#     return augmented_data


class LinearWarmUpCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, *, warmup_iters, max_learning_rate, min_lr, lr_decay_iters, **kwargs):
        self.warmup_iters = warmup_iters
        self.max_learning_rate = max_learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def _step_inner(self, steps):
        if steps < self.warmup_iters:
            return self.max_learning_rate * steps / self.warmup_iters
        elif steps < self.lr_decay_iters:
            return self.min_lr + 0.5 * (1.0 + np.cos((steps - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)*np.pi)) * (self.max_learning_rate - self.min_lr)
        else:
            return self.min_lr


def transform_img(img):
    # 处理图片
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # C, H, W
    img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
    # normalize
    normalized_img = img_tensor.float() / 255.0
    return normalized_img


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_img_torchvision(data):
    data['x'] = [transform(img.convert('RGB')) for img in data['image']]
    return data


label_mapping = {
    "nailong": 0,
    "emoji": 1,
    "anime": 2,
    "others": 3,
    "long": 4
}

def extract_datasets():
    ds = load_dataset("refoundd/NailongClassification", cache_dir="data", split="train")
    ds = ds.map(lambda x: {'label': label_mapping[x['label']]})
    ds = ds.map(transform_img_torchvision, remove_columns=['image'], batched=True)
    dataset = ds.train_test_split(test_size=0.2)
    return dataset

dataset = extract_datasets()


class NaiLongDataset(Dataset):
    def __init__(self, mode='train'):
        assert mode in ['train', 'test']
        self.dataset = dataset[mode]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]['x']
        label = self.dataset[idx]['label']
        return torch.tensor(item), torch.tensor(label)



class NaiLongTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self):
        # model = resnet18()
        # model.fc = torch.nn.Linear(model.fc.in_features, 2)
        # return model
        return convnext_base(pretrained=False, num_classes=5)
    
    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()
    
    def build_scheduler(self):
        return LinearWarmUpCosineAnnealingLR(self.optimizer, warmup_iters=self.config['warmup_iters'], max_learning_rate=self.config['max_learning_rate'], min_lr=self.config['min_lr'], lr_decay_iters=self.config['lr_decay_iters'])
    
    def build_dataloader(self, mode='train'):
        dataset = NaiLongDataset(mode="train")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_step(self, batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return self.criterion(self.model(x), y)
    
    def validate(self):
        self.logger.info("Validating...")
        self.model.eval()
        dataloader = self.build_dataloader(mode='test')
        acc = []
        f1 = [[], []]
        with torch.no_grad(): 
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                # print(f"Validation: {i}, {y}")
                y_hat = self.model(x)
                acc.append(torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y))
                f1[0].extend(y.cpu().tolist())
                f1[1].extend(torch.argmax(y_hat, dim=1).cpu().tolist())
            f1_scores = f1_score(f1[0], f1[1], average='macro')
        return {'acc': sum(acc) / len(acc), 'f1': f1_scores}


def main():
    config = {  # test
        "model": "convnext_tiny",
        "checkpoint_dir": "./checkpoints",
        "tensorboard_dir": "./tensorboard",
        "device": "cuda",
        "enable_cudnn_benchmark": True,
        "enable_amp": False,
        "learning_rate": 1,  # 启动lr_scheduler 这里必须是1
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "enable_compile": False,
        "weight_decay": 0.0,
        "max_steps": 5000,
        "max_grad_norm": 1.0,
        "save_every": 500,
        "gradient_accumulation_steps": 1,
        "warmup_iters": 500,
        "max_learning_rate": 1e-3,
        "min_lr": 1e-4,
        'lr_decay_iters': 1000
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['tensorboard_dir'], exist_ok=True)
    trainer = NaiLongTrainer(config)
    trainer.train()

if __name__ == "__main__":
    # 删除tensorboard下的文件, 但不删除文件夹
    for i in os.listdir('./tensorboard'):
        os.remove(os.path.join('./tensorboard', i))
    # 删除checkpoints下的文件
    for i in os.listdir('./checkpoints'):
        os.remove(os.path.join('./checkpoints', i))
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        
```

</p>
</details> 

### 数据增广

原始数据集只有两百多张图片, 这个时候无法避免的要做数据增广, 扩展 nailong 标签的数据, 这里因为是初版方案, 也没有非常精细的增广方案, 这里使用了以下几种方式(代码在如上train.py中):

- 给图片添加颜色遮罩
    让模型不要将遇到黄色的就判定为奶龙
- 在负样本中嵌入正样本
    很经典的增广数据的手法
- 图片轴对称
- 取图像的一半镜像翻转

### 训练

#### 参数搜索

虽然是个人小项目, 简单的参数搜索不能少, 继续上面写的 `trainer.py`, 我也写了一个简单的 `hyperparameter_seacher.py` 来搜索超参

<details><summary>hyperparameter_seacher.py</summary>
<p>

```python
from trainer import Trainer
import optuna


example_config = {
    "model": "convnext_tiny",
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
    "max_steps": 100,
    "max_grad_norm": 1.0,
    "save_every": 1000000,  # 不保存
    "gradient_accumulation_steps": 4
}

example_search_config = {
    'params': {
        "learning_rate": {
            "type": "float",
            "range": [1e-5, 1e-2],
            "log": True
        },
        "gradient_accumulation_steps": {
            "type": "int",
            "range": [1, 8],
            "log": False
        }
    },
    "if_save_info": False,
    "n_trials": 10
}

class HyperparameterSearcher:
    def __init__(self, config, trainer):
        assert isinstance(trainer, Trainer), "trainer must be an instance of Trainer"
        self.config = config
        self.trainer = trainer
        
    def objective(self, trial):
        search_params = self.config['params']
        
        for param_name, param_config in search_params.items():
            if param_config["type"] == "float":
                self.trainer.config[param_name] = trial.suggest_float(
                    param_name,
                    param_config["range"][0],
                    param_config["range"][1],
                    log=param_config.get("log", False)
                )
            elif param_config["type"] == "int":
                self.trainer.config[param_name] = trial.suggest_int(
                    param_name,
                    param_config["range"][0],
                    param_config["range"][1]
                )
            elif param_config['type'] == 'list':
                self.trainer.config[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['range']
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}, only support float and int")
        
        self.trainer.setup_training()
        self.trainer.train()
        metric = self.trainer.validate()
        if 'acc' not in metric:
            raise ValueError("metric must contain 'acc'")
        return -metric['acc']  # only support maximizing acc
    
    def search(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config['n_trials'])
        print("Best params:", study.best_params)
        print("Best value:", -study.best_value)
        if self.config['if_save_info']:
            study.trials_dataframe().to_csv("./output/optuna_results.csv")
        return study.best_params
    
def main():
    
    pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
```

</p>
</details> 

超参搜索需要 trainer 的配合, 使用了与模型无关的 optuna 来跑给定范围的超参值, 这个时候可以给trainer一个比较容易训练的超参设置(短的epoch等), 同时关闭保存模式

我也写了个简单的超参搜索的例子

<details><summary>hyperparameter_seacher使用案例</summary>
<p>

from example_trainer import Cifer10Trainer
from hyperparameter_seacher import HyperparameterSearcher

class Cifer10HyperparameterSearcher(HyperparameterSearcher):
    def __init__(self, config, trainer):
        super().__init__(config, trainer)


def main():
    search_config = {
        'params': {
            "learning_rate": {
                "type": "float",
                "range": [1e-5, 1e-2],
                "log": True
            },
            "gradient_accumulation_steps": {
                "type": "int",
                "range": [1, 8],
                "log": False
            }
        },
        "if_save_info": True,
        "n_trials": 10
    }

    trainer_config = {
        "model": "resnet18",
        "checkpoint_dir": "./checkpoints",
        "tensorboard_dir": "./tensorboard",
        "device": "cuda",
        "enable_cudnn_benchmark": True,
        "enable_amp": False,
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "enable_compile": False,
        "weight_decay": 0.05,
        "max_steps": 500,
        "max_grad_norm": 1.0,
        "save_every": 10000,  # large than max_steps, no save
        "gradient_accumulation_steps": 4,
        'batch_size': 32
    }
    trainer = Cifer10Trainer(trainer_config)
    searcher = Cifer10HyperparameterSearcher(search_config, trainer)
    best_params = searcher.search()
    print(best_params)

if __name__ == '__main__':
    main()

</p>
</details> 

> 搜索上面那个例子中的合适的超参数

在准备好这些后, 编写我们项目的超参搜索器

<details><summary>之后补充</summary>
<p>



</p>
</details> 

> 搜索出来的最佳超参是一个很长的小数, 四舍五入合适的位数即可

#### 第一次训练

在准备好后, 开始第一次训练

在较新的GPU下, 训练以前较小的模型可谓降维打击, 不到一个小时训练完毕

然而, 第一个问题出来了

acc很高, f1很低

编写测试代码:

<details><summary>test.py(非初版代码, 仅供参考)</summary>
<p>

```python
from sklearn.metrics import f1_score
import torch

from model import convnext_base
from PIL import Image
import numpy as np
from glob import glob
from torchvision import transforms

device = "cuda"
image_size = 224

model = convnext_base(pretrained=False, num_classes=5).to(device)
# model = resnet18()
# model.fc = torch.nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load('./checkpoints/best.pt', map_location=device)
model.load_state_dict(checkpoint['model'])

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_img(img):
    # 处理图片
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # C, H, W
    img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)
    # normalize
    normalized_img = img_tensor.float() / 255.0
    return normalized_img


def get_input_images(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    return torch.tensor(img).to(device).unsqueeze(0)

model.eval()

# 导出onnx
input_names = ["input"]
output_names = ["output"]
dynamic_axes = {
    "input": {0: "batch_size"},  # 输入的第一个维度是动态的
    "output": {0: "batch_size"}  # 输出的第一个维度是动态的
}
torch.onnx.export(model, torch.randn(1, 3, 224, 224).to(device), "model.onnx", input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)

with torch.no_grad():
    # image = torch.randn(1, 3, 256, 256)
    print(torch.softmax(model(get_input_images('1.jpg')), dim=1))
    input()
    print(torch.softmax(model(get_input_images('3.jpg')), dim=1))
    input()
    acc = []
    f1 = [[], []]
    
    
    for file in glob('./datasets/nailong/*'):
        y_hat, y = model(get_input_images(file)), torch.tensor([0]).to(device)
        acc.append(torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y))
        f1[0].append(y.cpu().tolist()[0])
        f1[1].append(torch.argmax(y_hat, dim=1).cpu().tolist()[0])
        

    # for file in glob('./datasets/cifer10/*'):
    #     y_hat, y = model(get_input_images(file)), torch.tensor([3]).to(device)
    #     acc.append(torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y))
    #     f1[0].append(y.cpu().tolist()[0])
    #     f1[1].append(torch.argmax(y_hat, dim=1).cpu().tolist()[0])

print(sum(acc) / len(acc))
f1_scores = f1_score(f1[0], f1[1], average='macro')
print(f1_scores)
```

</p>
</details> 


发现 acc 与 f1 的值非常低(百分之10到百分之20附近)

这个时候查看模型的输出, 发现模型的输出接近初始化的输出(softmax后), 模型没有怎么被训练

这个时候怀疑是训练代码出了问题, 然而 cifer10 的 训练并没有什么问题
检查数据增广代码, 查看增广后的图片, 发现增广做的不是很好, 正样本内嵌负样本没嵌好

在经过修改后, 重启训练
然而, 问题变成了, 模型的输出接近(0.5, 0.5)(二分类任务)

跟人讨论后, 认为是数据集难度太大, 检查表情包数据集, 都是一些分布与 nailong 数据差异很大的图片. 

> 非严格推理, 纯脑测
模型发现, 给一张新的图片预测 nailong 类, 还是其他类, 都会导致loss上升, 于是干脆摆烂乱猜,  最终的概率分布会输出数据分布, 经过数据增广后的数据恰好是两类 1:1, 模型退化成统计数据集了

导致这个的最直接原因是输入特征不够, 到图像分类就是模型找不到决定图片分类的模式

于是, 第一阶段的训练结束了

#### 第二次训练

在数据集作者不断的努力下, nailong 数据集有了一些完善, 主要的完善点在于: 
1. 新添更多 nailong
2. 不是二分类了, 新增了表情包分类, 动画分类等五分类, 不过不同类别的数据数量差异很大(两个数量级)
3. 加入了一些 corner case, 比如 藤田琴音等其他颜色为黄色的图像

因为第一次训练代码已经写好了, 改起来也不是很麻烦, 只需要换个数据集定义与读取. 作者的数据集放在 huggingface 上, 于是我们使用 datasets 进行读取.

> 我也不知道是不是我写的问题, datasets读起来很慢, dataloader 后, 会把 label 自动变成torch.tensor格式, 但是 n, c, h, w 格式的图片只会把 w 维度变成 torch.tensor 格式, 其他维度还是 List, 需要在 dataset 类定义的时候使用 __getitem__() 将数据提前变为 torch.tensor
> 然后不支持多线程读取(会卡住), 单线程读取读起来很慢, gpu 的 cuda 呈现尖刺状
> 然后, dataset 的读取**要先**读取 id 再读取 x 跟 label
> 没怎么用过 dataset, 这次属实是学到了

修改好后数据加载的代码后并注释掉先前的数据增广代码后(后续研究), 第二次训练开始了

这次结果好过头了
模型的 loss 收敛到了 $1e^{-5}$, acc跟f1更是到达了 $100\%$

使用测试代码简单测试, 发现在数据集的数据都能完美分类, 不在数据集的分类只要分不出是奶龙即可. 检查模型输出权重, 也没啥问题, 看起来是完美了?

然而 这张图还是给了模型一拳

<details><summary>图</summary>
<p>

![22](https://github.com/user-attachments/assets/3cb2b111-e01f-44dd-8e71-0509ab2bb6c0)

</p>
</details> 


他会识别成 nailong, 不过我觉得问题不大(确实有人把他抽象的认成 nailong)

### 部署

上面的 `test.py` 中 写了onnx导出的代码, 支持任意 batch 的输入(解锁了 n, c, h, w 的 n 维度)

简单编写onnx推理代码

<details><summary>onnx_inference.py</summary>
<p>

```python
# from torchvision import transforms
import onnxruntime as ort
from PIL import Image
import numpy as np

img_size = 224

# transform = transforms.Compose([
#     transforms.Resize((img_size, img_size)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def transform_img(img: Image, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = (img / 255 - mean) / std
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


label_mapping = {
    "nailong": 0,
    "emoji": 1,
    "anime": 2,
    "others": 3,
    "long": 4
}

reverse_label_mapping = {v: k for k, v in label_mapping.items()}

model_path = 'model.onnx'
session = ort.InferenceSession(model_path)

image_path = '3.jpg'
image = Image.open(image_path).convert("RGB")
# image = transform(image).unsqueeze(0).numpy()
image = transform_img(image)

# 运行推理
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: image})

# 获取分类结果
output = outputs[0]
predicted_class = np.argmax(output, axis=1)
predicted_label = reverse_label_mapping[predicted_class[0]]

print(f"Predicted class: {predicted_label}")

```

</p>
</details> 

> 训练的时候引入了 torchvision 的 transforms, 这里为了减少依赖, 选择手动实现, 有需要也可以自行取消注释并修改
