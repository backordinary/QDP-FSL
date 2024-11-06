import json
import os
import random
from collections import defaultdict

def split_train(input_jsonl, output_dir,time, test_ratio=0.2):
    """
    从输入的jsonl文件中读取数据，并按给定的比例划分数据集。

    参数:
        input_jsonl (str): 输入的jsonl文件路径
        output_dir (str): 输出文件夹路径，将在该文件夹中保存 data.jsonl, test.jsonl
        test_ratio (float): 测试集占比（默认为0.2）

    输出:
        保存train.jsonl, valid.jsonl, test.jsonl文件到指定的输出文件夹。
    """
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 创建存储数据的字典，以label为键
    data_by_label = defaultdict(list)

    # 读取输入jsonl文件
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            # 将每一行转换为JSON对象
            json_obj = json.loads(line.strip())
            label = json_obj['label']
            # 将相同label的对象放在一起
            data_by_label[label].append(json_obj)

    # 创建列表来存储train、valid、test数据
    train_data = []
    valid_data = []
    test_data = []

    # 按每个标签的数据进行划分
    for label, items in data_by_label.items():
        # 打乱数据顺序
        random.shuffle(items)

        # 计算每个数据集的大小
        total = len(items)
        test_size = int(total * test_ratio)
        train_size = total - test_size

        # 划分数据集
        test_data.extend(items[:test_size])
        train_data.extend(items[test_size:])

    # 定义保存的文件路径
    train_file = os.path.join(output_dir, f'{time}data.jsonl')
    test_file = os.path.join(output_dir, f'{time}test.jsonl')

    # 保存数据到jsonl文件
    def save_to_jsonl(data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_to_jsonl(train_data, train_file)
    save_to_jsonl(test_data, test_file)

    print(f"Data split completed:\nTrain: {len(train_data)}\nTest: {len(test_data)}")

# 示例调用
if __name__ == '__main__':
    input_jsonl = './data/Quantum/data.jsonl'  # 修改为实际输入文件路径
    for i in range(5) :
        output_dir = f'./data/Quantum/split_data{i}'  # 修改为实际输出文件夹路径
        split_train(input_jsonl, output_dir,time=0,test_ratio=0.2)
