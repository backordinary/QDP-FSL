from __future__ import absolute_import, division, print_function


import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from easyfsl.datasets import WrapFewShotDataset
from easyfsl.methods import FewShotClassifier
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fsl_text_dataset import FewShotTextDataset

logger = logging.getLogger(__name__)
def evaluate_one_task(
        model: FewShotClassifier,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
) -> Tuple[int, int]:
    """
    返回查询标签的正确预测数量和总预测数量。

    :param model: FewShotClassifier 模型实例
    :param support_images: 支持集图像，形状为 (n_support, 256)
    :param support_labels: 支持集标签，形状为 (n_support,)
    :param query_images: 查询集图像，形状为 (n_query, 256)
    :param query_labels: 查询集标签，形状为 (n_query,)
    :return: 正确预测数量和总预测数量
    """
    # 处理支持集
    model.process_support_set(support_images, support_labels)

    # 获取查询集的预测结果
    predictions = model(query_images).detach()

    # 计算正确预测的数量
    number_of_correct_predictions = (
        (torch.argmax(predictions, dim=1) == query_labels).sum().item()
    )

    # 返回正确预测的数量和查询集的总数量
    return number_of_correct_predictions, query_labels.size(0)

def evaluate(args, model, tokenizer,round_turn,time,tqdm_prefix: Optional[str] = None):

    if not args.isLocal :
        easy_eval_dataset = FewShotTextDataset(tokenizer, args, args.eval_data_file)
        eval_dataset = WrapFewShotDataset(easy_eval_dataset)
        with open(f'../data{time}/pkl/valid_dataset{round_turn}.pkl', 'wb') as f:
            pickle.dump(eval_dataset, f)
    else:
        with open(f'../data{time}/pkl/valid_dataset{round_turn}.pkl', 'rb') as f:
            eval_dataset = pickle.load(f)

    eval_sampler = TaskSampler(
        eval_dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, n_tasks=args.n_valid_per_epoch
    )
    n_workers = 4
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=eval_sampler.episodic_collate_fn,
    )



    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))

    total_predictions = 0
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        # We use a tqdm context to show a progress bar in the logs
        with tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
            ) in tqdm_eval:
                correct, total = evaluate_one_task(
                    model,
                    support_images.to(args.device),
                    support_labels.to(args.device),
                    query_images.to(args.device),
                    query_labels.to(args.device),
                )


                total_predictions += total
                correct_predictions += correct

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)
    return correct_predictions / total_predictions