对于自然语言处理来讲, LLM 实现了大一统, 干掉了所有传统自然语言处理
但在结构化数据建模, 建模依旧高度依赖特定的流水线, 不同的数据使用不同的算法(XGBoost, 随机森林, 不同的特征工程).
LimiX 旨在通过大型结构化数据基础模型, 来填补结构化数据的通用处理能力, 来实现通用人工智能

## 核心动机与研究定位

传统的表格处理方法 (包括深度表格网络和早期的表格基础模型) 局限于单一任务, 且面对新的数据集时需要微调.
LimiX 重新定义了这一范式: 将结构化数据建模变成对变量与缺失值的联合分布建模
仅需单一模型 无需梯度更新 无需微调 即可在推理阶段完成分类 回归 缺失值填补 数据值生成等任务


## 架构

### LimiX 拥有轻量 Transformer 结构

传统 Transformer 难以直接处理具有行列二维属性的表格数据, LimiX 将结构化数据映射为一组 "Sample-Feature Embeddings", 并并采用了由 12 层 Transformer Block 组成的非对称自注意力机制
在 特征维度(列) 进行两次注意力传递
在 样本维度(行) 进行一次注意力传递

#### 细节

标准 transformer  对于 扁平化序列会产生 $\mathcal{O}((NM)^2)$ 的计算复杂度, 对于较大规模的表格数据来说是不可接受的.

1. 特征维度级别注意力
    对于 第 $i$ 个样本的特征切片 $Z_{i, *, :} \in \mathbb{R}^{M \times d}$
    $$Q_{feat} = Z_{i}W_Q^{(f)}, \quad K_{feat} = Z_{i}W_K^{(f)}, \quad V_{feat} = Z_{i}W_V^{(f)}$$
    $$\text{Attn}_{feat}(Z_{i}) = \text{Softmax}\left(\frac{Q_{feat}K_{feat}^\top}{\sqrt{d_k}}\right)V_{feat}$$
    > 与一般的
2. 样本维度级别注意力
    对于第 $j$ 个特征的样本切片 $Z_{*, j, :} \in \mathbb{R}^{N \times d}$
    $$Q_{samp} = Z_{j}W_Q^{(s)}, \quad K_{samp} = Z_{j}W_K^{(s)}, \quad V_{samp} = Z_{j}W_V^{(s)}$$
    $$\text{Attn}_{samp}(Z_{j}) = \text{Softmax}\left(\frac{Q_{samp}K_{samp}^\top}{\sqrt{d_k}}\right)V_{samp}$$

加到一起

$$\tilde{\mathcal{Z}}_1 = \mathcal{Z} + \text{Attn}_{feat}(\text{LN}(\mathcal{Z}))$$

$$\tilde{\mathcal{Z}}_2 = \tilde{\mathcal{Z}}_1 + \text{Attn}_{samp}(\text{LN}(\tilde{\mathcal{Z}}_1))$$

$$\tilde{\mathcal{Z}}_3 = \tilde{\mathcal{Z}}_2 + \text{Attn}_{feat}(\text{LN}(\tilde{\mathcal{Z}}_2))$$

$$\mathcal{Z}_{out} = \tilde{\mathcal{Z}}_3 + \text{FFN}(\text{LN}(\tilde{\mathcal{Z}}_3))$$

> 两次注意力传递体现在这里 这里有两次的feat attn

最终复杂度变成了 $\mathcal{O}(NM^2 + MN^2)$

### LimiX 采用了 判别性特征编码 (Discriminative Feature Encoding)

针对不同表格的异构数据, LimiX 引入了可学习的低秩列标识符 (Low-rank Column Identifiers).
模型无需依赖固定的特征名称或位置, 就能在统一的高维空间中辨识和对齐不同来源的特征语义.

> 通过语义建模整列

#### 数据表示

1. 针对数据的embedding
    表格数据通常是二维的张量: $\mathcal{X} \in \mathbb{R}^{N \times M}$ 其中 $N$ 是样本数(行), $M$ 是特征数(列)
    对于表格中的任意单元格, 无论是连续还是离散先通过线性映射或者查表转换为维度为 $d$ 的向量
        
    > 想了想没找到合适的词 使用 ai 推荐的单元格, 实际上是$x_{i,j}$ 其中 $i$ 是第 i 个样本 $j$ 是第 j 个特征

2. 针对整列的embedding (低秩列标识符)
    引入一个可学习的低秩矩阵 $C \in \mathbb{R}^{M \times r}$ 其中 $r$ 为秩且满足 $r \ll d$。
    对于第 $j$ 列 提取其低维编码 $c_j \in \mathbb{R}^r$, 并通过线性变换矩阵 $W_c \in \mathbb{R}^{r \times d}$ 将其升维:
    $$ \tilde{c}_j = c_j W_c$$

3. 最终输出
    单元格最终的初始嵌入表示为基础特征与列标识符的加和:
    $$z_{i,j} = h_{i,j} + \tilde{c}_j$$

由此构成了模型的初始三维输入张量 $\mathcal{Z} \in \mathbb{R}^{N \times M \times d}$。

### LimiX 使用了 上下文作为条件的掩码预训练 (Context-Conditional Masked Pretraining)

在训练的时候, 使用同一数据集的一部分数据作为上下文输入, 用于建立该数据集的统计先验
另一部分数据作为 query, 某些数据的单元格会被随机mask. 根据给定的上下文, 预测这些被mask的内容

#### 细节

在训练中 数据集被划分为 上下文集 $\mathcal{X}_{ctx}$ 和查询集 $\mathcal{X}_{query}$

引入一个mask矩阵 $\mathcal{M} \in \{0, 1\}^{N_q \times M}$ 作用于 $\mathcal{X}_{query}$
如果$m_{i,j} = 1$，则单元格被替换为 `[MASK]` 标记. 可见部分记作 $\mathcal{X}_{query}^{\setminus \mathcal{M}}$

模型的预训练目标是最小化被掩码位置的负对数似然

$$\mathcal{L}(\theta) = -\sum_{(i,j) \in \mathcal{M}} \log P_\theta\left(x_{i,j} \mid \mathcal{X}_{ctx}, \mathcal{X}_{query}^{\setminus \mathcal{M}}\right)$$

对于连续变量 使用均方误差 (MSE) 作为高斯对数似然的代理:
$$\mathcal{L}_{cont} = \sum \|x_{i,j} - \hat{x}_{i,j}\|^2$$

对于分类变量 使用交叉熵:
$$\mathcal{L}_{cat} = -\sum x_{i,j} \log(\hat{p}_{i,j})$$

## 推理

推理阶段, LimiX 利用训练好的权重 $\theta$, 通过传入上下文样本, 利用模型自带的注意力score, 进行上下文学习.

1. 第一遍前向传播 提取注意力
     将所有的上下文样本 $\mathcal{X}_{ctx}$ 和测试样本输入模型. 在最后一层 $L$, 提取预测目标列 $y$ 与其他特征列的特征级注意力矩阵 $\mathcal{A}^{(f)}$, 以及测试样本与上下文样本间的样本级交叉注意力矩阵 $\mathcal{A}^{(s)}$.
    特征重要性可以形式化为:
    $$Score(feat_j) = \frac{1}{N}\sum_{i=1}^N \mathcal{A}^{(f)}_{i, y, j}$$
    > 预热 先捕捉数据集的关系 之前针对不同数据集的针对性建模就在这一步完成
2. 第二遍前向传播 条件推理
    根据 $Score(feat_j)$ 和样本级分数, 模型会对上下文进行加权过滤或重组, 到一个提纯后的定制化上下文 $\tilde{\mathcal{X}}_{ctx}$, 随后执行最终的条件预测:
    $$\hat{y}_{test} = \arg\max_y P_\theta(y \mid \tilde{\mathcal{X}}_{ctx}, x_{test})$$

这种机制天然赋予了 LimiX 极强的可解释性与因果推断能力

## 测试

在 kaggle 上进行测试
受限于环境 本文对于开源代码进行了修改
1. 取消了 flashattn 使用pytorch的标准混精度实现
2. 受限于显存 大幅度的降低了第一遍前向传播的上下文样本 这也是大幅影响模型效果的原因
3. 没有精调 没有指定数据类型


[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
Public Score: 0.80143
573/11597
> 此数据集用于测试


[Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2)
Private Score: 0.95257
Public Score: 0.95089
3167/4371

[Predicting Student Test Scores](https://www.kaggle.com/competitions/playground-series-s6e1)
Private Score: 8.92351
Public Score: 8.90245
1/4319

> 这场比赛已经结束, 我也不知道为什么score会这么高 第一名8.57273 遥遥领先
