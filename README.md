# BERT-NER

基于[transformers](https://github.com/huggingface/transformers)的训练框架。

* 对模块进行封装，支持灵活的修改。

* 对部署支持更好。

* 支持bert_ner、bert_distillation、electra等诸多任务的训练。

* 支持分布式训练、半精度浮点训练。

  

## Install

1. `pip install requirements.txt`

2. `pip install --no-cache-dir -i https://pypi.shannonai.com/root/stable/+simple/ --trusted-host pypi.shannonai.com shannon-preprocessor nlpc service-streamer`



## Train

* 框架说明：
  * 训练之前使用`dataset_readers`预生成训练数据，加快训练
  * `run`中进行训练参数初始化以及模型初始化
  * `run`调用`trainer`进行模型训练、调用`evaluator`进行模型测试
  * `trainer`调用`evaluator`进行模型测评估
* 修改：若需要修改框架进行不同任务训练，则至多需要在dataset_readers、run、trainers、evaluators中添加相应文件
  * 添加新的`dataset_reader`之后注意在`__init__.py`中做相应修改
  * 需要在`run`中完成`args`添加以及模型初始化
  * `trainers`中已有基类`trainer_base`，继承之后只需要重写`train、step、loss`函数即可
  * `evaluators`中已有基类`evaluator_base`，继承之后只需要重写`eval`函数即可
* 训练：examples中包含不同的task，以bert_ner为例
  * binarize：`examples/ner/binarize.sh`
  * train：`examples/ner/train.sh`
  * evaluate：`examples/ner/evaluate_ner.sh`

* 关于`models`
  * `bert_output_attn_score.py`：在`bert`的基础上修改使其输出softmax之前的`attention score`，便于bert_distillation的训练
  * `bert_mobile.py`：针对`mobilebert`对bert进行修改，增加`bottleneck`结构
  * `bert_mobile_config.py`：`mobilebert`的相应`BertConfig`类，增加`intra_size`和`num_ffn`参数



## Deploy

在`bert_ner/deploy`中封装有部署做inference时调用的类。
调用方法可以参考`tests/deploy`



## TODO

1. 支持CRF
2. 仿照allennlp的register和fairseq的argparser对项目进行模块化。
3. 支持interative
4. 完善requirements如对transformer的依赖程度的测试。
5. 支持tensorRT部署。
