from typing import Iterator, List, Dict
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
import argparse

from allennlp.data import Instance
from allennlp.data.fields import TextField,LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import BertClassifier
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    # 文件路径：数据目录， 缓存目录
    parser.add_argument("--train_data_dir",
                        default='/home/xyf/桌面/Disk/NLP语料/文本分类/ChnSentiCorp/ChnSentiCorp情感分析酒店评论/train.tsv',
                        type=str)
    parser.add_argument("--dev_data_dir",
                        default='/home/xyf/桌面/Disk/NLP语料/文本分类/ChnSentiCorp/ChnSentiCorp情感分析酒店评论/dev.tsv',
                        type=str)

    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--log_dir",
                        default='logs',
                        type=str,
                        help="日志目录，主要用于 tensorboard 分析")

    parser.add_argument("--bert_model_dir",
                        default='/home/xyf/models/chinese/bert/pytorch/bert-base-chinese',
                        type=str)

    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # 训练参数
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--dev_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for dev.")
    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for test.")

    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # optimizer 参数
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam 的 学习率")
    # 梯度累积
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--print_step',
                        type=int,
                        default=50,
                        help="多少步进行模型保存以及日志信息写入")

    parser.add_argument("--early_stop", type=int, default=10, help="提前终止，多少次dev loss 连续增大，就不再训练")

    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu 的设备id")

    config = parser.parse_args()
    return config

config = get_args()
print(config)


torch.manual_seed(1)
class ClassifierDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, PretrainedTransformerIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = PretrainedTransformerTokenizer(model_name=config.bert_model_dir)
        self.token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(
            model_name=config.bert_model_dir,max_length=config.max_seq_length)}

    def text_to_instance(self, sentence: str, label: str = None) -> Instance:
        word_tokens = self.tokenizer.tokenize(sentence)
        sentence_field = TextField(word_tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        f = pd.read_csv(file_path, sep='\t')
        for index,row in f.iterrows():
            label = str(row[1])
            sentence = row[0]
            if label is np.nan or sentence is np.nan: continue
            yield self.text_to_instance(sentence, label)


reader = ClassifierDatasetReader()
train_dataset = reader.read(cached_path(config.train_data_dir))
validation_dataset = reader.read(cached_path(config.dev_data_dir))
# tokens索引使用预训练语言模型, labels索引使用了这个vocab
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

model = BertClassifier(vocab=vocab, bert_model=config.bert_model_dir,)


if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)


iterator = BucketIterator(batch_size=config.train_batch_size)
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=config.early_stop,
                  num_epochs=config.num_train_epochs,
                  cuda_device=cuda_device)
trainer.train()














