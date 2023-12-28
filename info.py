import os
import pandas as pd
from collections import defaultdict

COUNT_NUM = defaultdict(int)

def combine_file_paths(folder):
    """ Get all the file paths from dataset.
    """
    all_files = []
    for path, _, file_ in os.walk(folder):
        for file_name in file_:
            all_files.append(os.path.join(path, file_name))
    print(all_files)
    return all_files

def count_num_types(file_list):
    """ Calculate the number of types.
    """
    print(f">>> Totally {len(file_list)} dataset. (Delete iris.csv since it is not Software Defect Prediction Dataset.)\n")
    maxnum = 0
    for file in file_list:
        data = pd.read_csv(file)
        columns = data.columns
        label = data[columns[-1]]
        label = label.tolist()
        for item in label:
            COUNT_NUM[item] += 1
        if max(set(label)) > maxnum:
            maxnum = max(set(label))
        print(f">> {file} has {len(columns[-21:-1])} features, {len(set(label))} different bugs. And the whole number of it is {data.shape[0]}.")
        print(f"> Unique labels are {set(label)}.\n")
    # 计算所有样本数目
    all_num = 0
    for pair in COUNT_NUM.items():
        all_num += pair[1]
    print(f"All dataset's num is {all_num}.")
    print(f"Each type's num is {COUNT_NUM}.")
    print(f"So we can design a multiple classifier (num = {maxnum + 1}).")

    all_weights = [0 for i in range(maxnum + 1)]
    for key, value in COUNT_NUM.items():
        all_weights[key] = all_num*(1/value)
    print(f"So we can give them the following weights -- {all_weights}")
        

if __name__ == "__main__":
    folder_path = 'dataset'
    file_list = combine_file_paths(folder_path)
    # print(count_num_types(file_list))
    count_num_types(file_list)
    