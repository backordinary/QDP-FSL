import logging
import os
import pickle
import random
import numpy as np
import torch
from easyfsl.datasets import WrapFewShotDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler

import pandas as pd
from tqdm import tqdm
from fsl_text_dataset import FewShotTextDataset

logger = logging.getLogger(__name__)

def sample_support_set(train_dataset, n_way, n_shot):
    # 获取去重后的标签
    unique_labels = list(set(train_dataset.labels))
    # 随机选择 n_way 个类
    classes = random.sample(unique_labels, n_way)

    # 从每个类中选择 n_shot 个样本作为支持集
    support_indices = []
    for cls in classes:
        class_indices = [i for i, label in enumerate(train_dataset.labels) if label == cls]
        if len(class_indices) < n_shot:
            raise ValueError(f"Not enough samples for class {cls}: required {n_shot}, available {len(class_indices)}")

        support_indices.extend(random.sample(class_indices, n_shot))

    return support_indices

def test(args, model, tokenizer, round_turn, time):
    if not args.isLocal:
        eval_dataset = WrapFewShotDataset(FewShotTextDataset(tokenizer, args, args.test_data_file))
        train_dataset = WrapFewShotDataset(FewShotTextDataset(tokenizer, args, args.train_data_file))
        with open(f'../data{time}/pkl/test_dataset{round_turn}.pkl', 'wb') as f:
            pickle.dump(eval_dataset, f)
        with open(f'../data{time}/pkl/train_dataset{round_turn}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
    else:
        with open(f'../data{time}/pkl/test_dataset{round_turn}.pkl', 'rb') as f:
            eval_dataset = pickle.load(f)
        with open(f'../data{time}/pkl/train_dataset{round_turn}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)

    # 使用普通的 SequentialSampler
    eval_sampler = SequentialSampler(eval_dataset)
    n_workers = 4
    eval_loader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # 每次加载一个查询样本
        num_workers=n_workers,
        pin_memory=True,
    )

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for query_sample in tqdm(eval_loader, desc="Evaluating"):
            # 每次从测试集中获取一个查询样本
            query_image, query_label = query_sample

            # 动态采集支持集
            support_indices = sample_support_set(train_dataset, args.n_way, args.n_shot)

            # 获取支持集的样本和标签
            support_images = torch.stack([train_dataset[idx][0] for idx in support_indices])
            support_labels = torch.tensor([train_dataset.labels[idx] for idx in support_indices])

            # 确保将数据转为torch.Tensor并移动到设备上
            support_images = support_images.to(args.device)
            support_labels = support_labels.to(args.device)
            query_image = query_image.to(args.device)
            query_label = query_label.to(args.device)

            # 处理支持集并进行预测
            model.process_support_set(support_images, support_labels)
            predictions = model(query_image).detach()
            predicted_class = torch.argmax(predictions, dim=1).cpu().numpy()[0]

            true_labels.append(query_label.cpu().numpy())
            predicted_labels.append(predicted_class)

    # 计算指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    auc = roc_auc_score(true_labels, predicted_labels, average='weighted')

    print(f"Accuracy for model: {100 * accuracy:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 Score: {100 * f1:.2f}%")
    print(f"ROC AUC Score: {100 * auc:.2f}%")

    results_df = pd.DataFrame({
        'True Labels': true_labels,
        'Predicted Labels': predicted_labels
    })
    results_df.to_csv(f'../data{time}/csv/{time}results{round_turn}.csv', index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [accuracy, precision, recall, f1, auc]
    })
    metrics_df.to_csv(f'../data{time}/csv/{time}metrics{round_turn}.csv', index=False)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc
    }
    return metrics
