# Ancient Chinese Translator: 将现代文翻译成古文的翻译器

该项目是为了训练一个能够完成白话文到古文的翻译。开始这个项目一方面是为了好玩，另一方面也是作为使用
pytorch-lightning这个框架的一个测试。


## 训练环境

* torch >= 1.6.0
* pytorch-lightning >= 1.2.10

## 数据来源

使用的语料库为：https://github.com/bangboom/classical-chinese

通过将该语料库的所有书籍组合成: 白话文 -> 古文的格式进行训练，共有语料对890521条。


## 模型信息

本项目尝试了GRU和transformer两个模型，其中transformer是我直接对着论文《Attention is All you need》
写的，在实现上不完全与目前的主流实现对齐，而且从实验中也可以发现目前的实现上有一定的瑕疵，但考虑到该
项目的目的不是为了训练出一个足够好的翻译模型，而是用seq2seq的方式进行白话文到古文翻译的尝试，以及试用
pytorch-lightning框架，因此不再对该问题进行进一步的排查。

从实验结果来看，transformer的表现是远好于GRU的，并且transformer在encoder和decoder层数增多的情况下，
生成的结果上会更优一样（指标上没有明显变化，仅主观感受）。

## 训练信息

该项目最终使用的模型为transformer，hiddin_size和word_embedding都为256，ffn的hidden_size为256*4，encoder和decoder层数都为6，按字对进行tokenize，最终在测试集上的loss为。

## 效果

### Demo

[在线演示](http://39.105.30.61:5000/)

### 生成结果

从生成效果来看，在输入的描述偏书面语的情况下能有不错的表现，如：

> 今年三月，我和我的朋友一起出门游玩。<br/>
> 今年三月，与臣友同出游。

而且有时标点符号也能影响生成的语气，如：

> 就你话多。<br/>
> 即汝多言。

> 就你话多！<br/>
> 汝何言多！

但一旦表达比较接近现代用语，效果不太尽如人意了，如：

> 今天一起去吃火锅。<br/>
> 今日共食之。

## 项目使用说明

### 生成数据

### 训练模型

### 模型生成测试

### 模型服务部署
