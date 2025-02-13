> 本文存在大量口齿不清

deepseek-r1持续火热, 已有大量的复现训练过程

1. [open-r1](https://github.com/huggingface/open-r1)
2. [mini-deepseek-r1](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)
3. [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
4. [open-thoughts](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
5. [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
6. [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
7. [RAGEN](https://github.com/ZihanWang314/RAGEN)
8. [unsloth r1-reasoning](https://unsloth.ai/blog/r1-reasoning)
9. [oat-zero](https://github.com/sail-sg/oat-zero)
10. [Logic-RL](https://github.com/Unakar/Logic-RL)
11. [deepseek-r1-gsm8k](https://github.com/Mryangkaitong/deepseek-r1-gsm8k)

具体效果可以看具体的仓库与其中的论文或者博客, 这里主要想法是, 是否可以使用这一训练范式, 来为传统强化学习任务, 带来可对人带来参考的思维过程

例如, 希望给围棋使用此范式, 一个可能的 pipeline 是:
1. sft, 为模型带入围棋知识
    > 我认为是必要的, 因为通用型大模型此类知识较少, 如果没有的话, 搜索解空间太大, 不容易搜索到正确的 token, 可以使用其他强力模型或者程序进行辅助构建
2. 设定一个通用格式, 继续sft, 让模型输出正确格式(非必要, 在上述复现中, 无需特意sft也可以让模型输出正确格式)
3. rule based RL
    > 考虑到围棋中间的reward非常难量化, 可以使用专业围棋模型进行反馈
    > 让大模型进行一定的思考, 输出一个答案, 之后与围棋模型进行比对, 前几选 reward +1, 其他 reward -1, 或者更进一步的, 使用围棋模型的 logits
    > 如果是简单的任务, 则可以通过env反馈奖励(与r1相同), 参考 [RAGEN](https://github.com/ZihanWang314/RAGEN)


