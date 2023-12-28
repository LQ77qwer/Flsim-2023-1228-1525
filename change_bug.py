import pandas as pd

# 读取CSV文件
data = pd.read_csv(r'merged_dataset.csv')

# 将 'bug' 列中非零值转换为1
data['bug'] = data['bug'].apply(lambda x: 1 if x != 0 else 0)
data.to_csv('new_merged_dataset.csv', index=False)
