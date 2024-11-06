from __future__ import absolute_import, division, print_function

import logging
import os
import pickle

import torch
from easyfsl.datasets import WrapFewShotDataset
from easyfsl.methods import FewShotClassifier
from easyfsl.samplers import TaskSampler
from torch import nn
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from statistics import mean

from tqdm import tqdm

from Evaluate_FSL import evaluate
from fsl_text_dataset import FewShotTextDataset
logger = logging.getLogger(__name__)


# 一轮训练函数
def training_epoch(
        model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer,LOSS_FUNCTION,DEVICE
):
    all_loss = []
    model.train()

    with tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            classification_scores = model(query_images.to(DEVICE))

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

def train(args, model, tokenizer,round_turn,time):
    """ Train the model """
    if not args.isLocal :

        train_dataset = WrapFewShotDataset(FewShotTextDataset(tokenizer, args, args.train_data_file))

        with open(f'../data{time}/pkl/train_dataset{round_turn}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
    else:
        with open(f'../data{time}/pkl/train_dataset{round_turn}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
    train_sampler = TaskSampler(
        train_dataset, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, n_tasks=args.n_tasks_per_epoch
    )
    n_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )



    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))


    # 定义损失函数和优化器
    # LOSS_FUNCTION = GHM_Loss(bins=10, alpha=0.75)
    LOSS_FUNCTION = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    # weight_decay用于防止过拟合
    train_optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=args.weight_decay
    )
    # 学习多少个epoch后调整学习率
    scheduler_milestones = [2, 5, 9, 14]
    scheduler_gamma = args.scheduler_gamma
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    best_acc = 0.0
    epochs_without_improvement = 0  # Early Stopping Counter

    for epoch in range(args.num_train_epochs):
        logger.info("Turn %d is training now---------------------",epoch)
        average_loss = training_epoch(model, train_loader, train_optimizer,LOSS_FUNCTION,args.device)
        logger.info("average_loss is :%s",average_loss)
        average_acc = evaluate(args, model, tokenizer,round_turn,time=time,tqdm_prefix='Validating')
        logger.info("average_acc is :%s",average_acc)

        # Check for improvement
        if average_acc > best_acc + args.min_delta:
            best_acc = average_acc
            epochs_without_improvement = 0
            logger.info("  "+"*"*20)
            logger.info("  Best acc:%s",round(best_acc,4))
            logger.info("  "+"*"*20)
            # Save best model checkpoint
            output_dir = f'../data{time}/checkpoint-best-acc{round_turn}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))
            logger.info("Model improved. Saving new best model to %s", output_dir)

        else:
            epochs_without_improvement += 1
            logger.info("No improvement. Early stopping patience: %d/%d",
                        epochs_without_improvement, args.early_stopping_patience)

        # Early stopping condition
        if epochs_without_improvement >= args.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break
        train_scheduler.step()