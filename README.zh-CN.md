# DIYBert

[English](./README.md)

本项目是对 [Texar example](https://github.com/asyml/texar-pytorch/tree/master/examples/bert)进行的简化和修改，便于对连接在BERT下游的模型结构进行添加、修改，便于自定义数据处理类，减少不常用代码块。

# Texar的优点:

- 加载模型进行训练之前，所有数据处理过程都已完成并保存在pickle文件中，训练时直接从pockle文件读入BERT需要的输入即可。
- 模型和数据处理的超参数设置分别集中保存在config_model.py和config_data.py文件中便于多次调参
- 基于pytorch便于在训练时即时得到模型每个计算层的输出，便于调试。

# 本项目主要功能:

- utils/data_utils.py定义**DiyProcessor**类，重载*_create_examples*方法能够灵活直接对多种不同任务的文本数据进行处理。
- main.py 定义**DiyBert**类支持在加载原有BERT Transformer-Encoder的基础上灵活设置下游模型结构。（目前已有代码仅支持分类任务，但已提供平台便于扩展到其他任务）
- main.py 重定义*_compute_loss*函数，使得训练过程更稳定
- 修改config_data.py 和config_model.py 保存所有需要的超参数信息。每次运行前打印所有超参数设置信息，便于调参实验进行对比
- 提供API加载本地已有预训练模型，无需重复下载

# 使用:

- 训练之前先要把文本数据转化为BERT要求的输入保存在data_dir目录下

  ```
  python prepare_data.py  \
      --max-seq-length=512 \
      --pretrained-model-name=bert-base-cased \
      --data-dir=data/MRPC \
      --LMroot=D:/LM \
  ```

- 开始训练后，终端会显示信息

  ```
  python main.py \
      --output-dir=models/ \
      --pretrained-model-name=bert-base-cased \
      --LMroot=D:/LM \
      --do-train \
      --do-eval \
      --do-test
  ```

- 解压MRPC.zip文件运行MRPC句子对分类例子.

# Lisence
[MIT](./LISENCE)

***

Forks and Stars are appreciated :blush: ~
