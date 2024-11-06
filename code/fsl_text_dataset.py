import logging
import torch
from torch.utils.data import Dataset
import json
from transformers import RobertaModel, RobertaConfig
from input_features import InputFeatures

logger = logging.getLogger(__name__)

def convert_examples_to_features(js, tokenizer, args):
    # 将代码文本转换为 tokens
    code = ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['label'])


class FewShotTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        """
        FewShotTextDataset 初始化，加载数据并计算嵌入。
        """
        self.examples = []
        self.labels = set()


        # 处理数据文件中的每一行
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                # feature = convert_examples_to_features(js, tokenizer, self.model, args)
                feature = convert_examples_to_features(js, tokenizer, args)
                self.examples.append(feature)
                self.labels.add(js['label'])  # 收集标签

        # 打印一些样例
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                #
                # # 打印嵌入的前5个向量，每个向量的前10个值
                # embedding_preview = example.input_embeddings[:5, :10].cpu()  # 转为 CPU 便于打印
                # logger.info("embedding preview (first 5 embeddings, first 10 dimensions):\n{}".format(embedding_preview))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """
        返回处理后的 嵌入和标签，供模型直接使用。
        """
        # return self.examples[i].input_embeddings, self.examples[i].label
        return torch.tensor(self.examples[i].input_ids, dtype=torch.long),self.examples[i].label

    def get_labels(self):
        """
        返回数据集中所有唯一标签。
        """
        return list(self.labels)
