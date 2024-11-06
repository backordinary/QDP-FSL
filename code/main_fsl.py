from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
from easyfsl.methods import FEAT
from easyfsl.modules import resnet12, MultiHeadAttention
from torch import nn
import torch.multiprocessing as mp
from transformers import (
    RobertaConfig, RobertaTokenizer, RobertaModel)

from Train_FSL import train
from Test_FSL import test

from data_split import split_data
from split_train import split_train
logger = logging.getLogger(__name__)

class CustomBackbone(nn.Module):
    def __init__(self, original_backbone):
        super(CustomBackbone, self).__init__()
        self.original_backbone = original_backbone

    def forward(self, x):
        # 获取 CodeBERT 的输出：last_hidden_state
        output = self.original_backbone(x).last_hidden_state
        # 对输出展平，确保每个样本是一个 1 维的特征向量
        return output[:, 0, :]  # 提取 [CLS] token 的嵌入作为句子的表示
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--scheduler_gamma", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")

    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help="Number of epochs to wait for improvement before stopping.")
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help="Minimum change to qualify as an improvement.")


    parser.add_argument('--n_way',  type=int, default=2)
    parser.add_argument('--num_folds',  type=int, default=4)
    parser.add_argument('--num_times',  type=int, default=10)
    parser.add_argument('--n_shot',  type=int, default=3)
    parser.add_argument('--n_query',  type=int, default=2)
    parser.add_argument('--n_tasks_per_epoch',  type=int, default=500)
    parser.add_argument('--n_valid_per_epoch',  type=int, default=100)
    parser.add_argument('--n_test_per_epoch',  type=int, default=100)
    parser.add_argument('--isLocal',  action='store_true')


    return parser
def main():

    # 设置多进程启动方式为 spawn
    mp.set_start_method('spawn', force=True)
    args = read_args().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1

    args.device = device


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name_or_path, local_files_only=True)
    config.num_labels=2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name, local_files_only=True)


    logger.info("Training/evaluation parameters %s", args)
    tot_dataset = '../tot_data.jsonl'
    accuracys_on_model = []
    precisions_on_model = []
    recalls_on_model =[]
    f1s_on_model =[]
    aucs_on_model =[]
    for time in range(args.num_times):
        if args.do_train :
            logger.info(f"Starting training time {time + 1}/{args.num_times}")
            prefix_time = f'../data{time}'
            split_train(input_jsonl=tot_dataset,output_dir=prefix_time,time = time,test_ratio=0.2)
            train_data_file = os.path.join(prefix_time, f'{time}data.jsonl')# 修改为实际输入文件路径

            split_output_dir = os.path.join(prefix_time,f'fold')
            if not os.path.exists(os.path.join(prefix_time,f'fold')):
                os.makedirs(os.path.join(prefix_time,f'fold'))
            if not os.path.exists(os.path.join(prefix_time,f'pkl')):
                os.makedirs(os.path.join(prefix_time,f'pkl'))
            split_data(train_data_file, split_output_dir,time=time,num_folds=args.num_folds)  # Assuming this function supports k-fold splitting

            for fold in range(args.num_folds):
                logger.info(f"Starting training fold {fold + 1}/{args.num_folds}")

                attention_module = MultiHeadAttention(8, config.hidden_size, 640, 640).to(args.device)
                codebert_backbone = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
                custom_backbone = CustomBackbone(codebert_backbone)
                model = FEAT(
                    backbone=custom_backbone,
                    attention_module=attention_module
                ).to(args.device)


                # Set training and evaluation files
                train_data_file = os.path.join(split_output_dir, f'{time}train_fold{fold}.jsonl')
                eval_data_file = os.path.join(split_output_dir, f'{time}valid_fold{fold}.jsonl')


                args.train_data_file = train_data_file
                args.eval_data_file = eval_data_file

                # Training
                train(args, model, tokenizer, round_turn=fold,time=time)

        if args.do_test:
            logger.info(f"Starting Testing time {time + 1}/{args.num_times}")
            prefix_time = f'../data{time}'
            args.train_data_file = os.path.join(prefix_time,f'{time}data.jsonl')
            args.test_data_file = os.path.join(prefix_time,f'{time}test.jsonl')

            if not os.path.exists(os.path.join(prefix_time,f'csv')):
                os.makedirs(os.path.join(prefix_time,f'csv'))

            accuracy = []
            precision = []
            recall =[]
            f1 =[]
            auc =[]
            for fold in range(args.num_folds):
                logger.info(f"Starting Testing fold {fold + 1}/{args.num_folds}")

                attention_module = MultiHeadAttention(8, config.hidden_size, 640, 640).to(args.device)
                codebert_backbone = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
                custom_backbone = CustomBackbone(codebert_backbone)

                model = FEAT(
                    backbone=custom_backbone,
                    attention_module=attention_module
                ).to(args.device)

                # Testing
                output_dir = os.path.join(prefix_time,f'checkpoint-best-acc{fold}/model.bin')
                model.load_state_dict(torch.load(output_dir))
                model.to(args.device)
                metric_one =  test(args, model, tokenizer, round_turn=fold,time=time)
                # 将每个折叠的指标添加到列表中
                accuracy.append(metric_one['accuracy'])
                precision.append(metric_one['precision'])
                recall.append(metric_one['recall'])
                f1.append(metric_one['f1_score'])
                auc.append(metric_one['roc_auc'])
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                'Value': [sum(accuracy) / args.num_folds, sum(precision) / args.num_folds, sum(recall) / args.num_folds, sum(f1) / args.num_folds, sum(auc) / args.num_folds]
            })
            metrics_df.to_csv(f'../data{time}/csv/{time}metrics_on_average.csv', index=False)
            accuracys_on_model.append(sum(accuracy) / args.num_folds)
            precisions_on_model.append(sum(precision) / args.num_folds)
            recalls_on_model.append(sum(recall) / args.num_folds)
            f1s_on_model.append(sum(f1) / args.num_folds)
            aucs_on_model.append(sum(auc) / args.num_folds)
    if args.do_test:
        accuracy_mean = sum(accuracys_on_model) / args.num_times
        precision_mean = sum(precisions_on_model) / args.num_times
        recall_mean = sum(recalls_on_model) / args.num_times
        f1_mean = sum(f1s_on_model) / args.num_times
        auc_mean = sum(aucs_on_model) / args.num_times
        metrics_on_model_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [accuracy_mean,precision_mean,recall_mean,f1_mean,auc_mean]
        })
        metrics_on_model_df.to_csv(f'../metrics_on_average_tot.csv', index=False)

if __name__ == "__main__":
    main()