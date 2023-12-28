import pandas as pd
import os

# 设置文件夹路径
folder_path = 'dataset'

# 获取文件夹中所有CSV文件的列表
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 初始化一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()

# 遍历文件列表，读取每个文件并追加到merged_data中
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    merged_data = merged_data.append(data, ignore_index=True)

# 保存合并后的数据到新的CSV文件
merged_data.to_csv('merged_dataset.csv', index=False)
