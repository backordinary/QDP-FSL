import os
import pandas as pd

import json
def remove_comments_from_code(code):
    """
    移除Python代码中的注释内容，包括行注释和块注释。

    参数:
        code (str): Python代码内容。

    返回:
        str: 移除了注释的代码。
    """
    import re
    # 移除行注释（#开头的注释）
    code = re.sub(r'#.*', '', code)

    # 移除块注释（'''或"""包裹的注释）
    code = re.sub(r'\'\'\'(.*?)\'\'\'', '', code, flags=re.DOTALL)
    code = re.sub(r'\"\"\"(.*?)\"\"\"', '', code, flags=re.DOTALL)

    # 移除多余的空行，将多个换行替换为一个
    code = re.sub(r'\n\s*\n', '\n', code)
    code = code.strip()
    # 去除首尾空白字符后，检查代码内容是否为空
    if code == '':
        return None
    # 返回移除注释后的代码
    return code

def read_python_files(root_dir=None, output_jsonl=None):
    """
    遍历data文件夹下的所有子文件夹，读取子文件夹中的.py文件，并根据所在子文件夹打标签。
    过滤掉注释内容，并保持换行符不变，最后将结果保存到Excel文件中。
    如果提供了input_excel的路径，则直接读取该xlsx文件并返回数据。

    参数:
        root_dir (str, 可选): 根目录路径，默认为当前工作目录下的data文件夹。
        output_excel (str, 可选): 如果提供，将结果保存为指定的Excel文件路径。

    返回:
        pd.DataFrame: 包含文件路径、文件内容和标签的DataFrame。
    """
    # 如果未提供root_dir，默认使用当前工作目录下的data文件夹
    if root_dir is None:
        root_dir = os.path.join(os.getcwd(), "data")

    # 检查 data 文件夹是否存在
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The directory '{root_dir}' does not exist.")

    # 创建一个空的列表，存储文件信息
    file_data = []

    # 用于保存唯一的class_names和其对应的数值标签
    class_name_to_label = {}

    # 遍历data文件夹下的所有子文件夹
    for foldername, subfolders, filenames in os.walk(root_dir):
        # 确保是在子文件夹中的文件，而不是根文件夹
        if foldername == root_dir:
            continue  # 跳过根文件夹data

        # 获取当前子文件夹的名字（作为标签）
        class_name = os.path.basename(foldername)

        # 如果class_name还没有对应的数值标签，则分配一个新的数值标签
        if class_name not in class_name_to_label:
            class_name_to_label[class_name] = len(class_name_to_label)  # 使用当前字典长度作为数值标签

        # 获取该class_name对应的数值标签
        label = class_name_to_label[class_name]

        # 遍历当前子文件夹中的所有文件
        for filename in filenames:
            if filename.endswith(".py"):
                # 构建完整文件路径
                file_path = os.path.join(foldername, filename)

                # 读取文件内容
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()

                    # 移除注释内容
                    file_content = remove_comments_from_code(file_content)

                    # 如果处理后的文件内容为空，则跳过此文件
                    if file_content is None:
                        print(f"Skipping empty file: {file_path}")
                        continue

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    file_content = None

                # 将文件路径、文件内容和标签保存到列表中
                file_data.append({
                    'file_path': file_path,
                    'content': file_content,
                    'class_names': class_name,  # 保存文件夹名作为class_name
                    'label': label  # 保存数值标签
                })

    # 将数据转换为DataFrame
    df = pd.DataFrame(file_data)

    # 如果提供了jsonl文件路径，则保存结果为jsonl文件
    if output_jsonl:
        # 获取文件夹路径
        output_dir = os.path.dirname(output_jsonl)

        # 如果文件夹不存在，创建该文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 将数据保存到jsonl文件中
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json_line = {"code": row['content'], "label": row['label']}
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

        print(f"Data saved to {output_jsonl}")


    return df



if __name__ == '__main__':
    # load_data()
    read_python_files(root_dir='../source',output_jsonl='../tot_data.jsonl')