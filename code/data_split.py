import json
import os
import random
from collections import defaultdict

def split_data(input_jsonl, output_dir,time, num_folds=4):
    """
    从输入的jsonl文件中读取数据，并按k折交叉验证的方式划分数据集。

    参数:
        input_jsonl (str): 输入的jsonl文件路径
        output_dir (str): 输出文件夹路径，将在该文件夹中保存训练和验证集文件
        num_folds (int): 折数（默认为4）

    输出:
        保存训练和验证集文件到指定的输出文件夹。
    """
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 创建存储数据的字典，以label为键
    data_by_label = defaultdict(list)

    # 读取输入jsonl文件
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            label = json_obj['label']
            data_by_label[label].append(json_obj)

    # 创建存储每个折的数据
    folds = [[] for _ in range(num_folds)]

    # 将每个标签的数据分到各个折中
    for label, items in data_by_label.items():
        random.shuffle(items)
        fold_size = len(items) // num_folds
        for i in range(num_folds):
            start = i * fold_size
            end = start + fold_size if i < num_folds - 1 else len(items)
            folds[i].extend(items[start:end])

    # 保存每个折的数据集
    for fold in range(num_folds):
        train_data = []
        valid_data = []

        # 组合其他折作为训练集
        for i in range(num_folds):
            if i != fold:
                train_data.extend(folds[i])

        # 当前折作为验证集
        valid_data.extend(folds[fold])

        # 保存数据到jsonl文件
        def save_to_jsonl(data, filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        save_to_jsonl(train_data, os.path.join(output_dir, f'{time}train_fold{fold}.jsonl'))
        save_to_jsonl(valid_data, os.path.join(output_dir, f'{time}valid_fold{fold}.jsonl'))

    print(f"Data split completed into {num_folds} folds.")

# 示例调用
if __name__ == '__main__':
    input_jsonl = '../data/data.jsonl'  # 修改为实际输入文件路径
    output_dir = '../data/k_fold'  # 修改为实际输出文件夹路径
    split_data(input_jsonl, output_dir)
