import numpy as np
import pandas as pd
import os


from info import combine_file_paths

class LoadDataset(object):
    def __init__(self, dataSetPath) -> None:
        self.data_path = dataSetPath

        self.X = []
        self.y = []
        self.rawy = []

        self.train_data = None
        self.train_label = None
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self.construct()
    
    def construct(self):
        """ construct the dataset
        """
        # self.dataset_name = self.data_path.split('\\')[1]
        self.dataset_name = os.path.basename(self.data_path)
        whole_dataset = pd.read_csv(self.data_path)
        columns = whole_dataset.columns
        # standardize
        for i in range(-21, -1):
            MAX = np.max(whole_dataset[columns[i]])
            MIN = np.min(whole_dataset[columns[i]])
            whole_dataset[columns[i]] = (whole_dataset[columns[i]] - MIN) / (MAX - MIN)
        whole_dataset = whole_dataset.values.tolist()
        for _,item in enumerate(whole_dataset):
            # print(item)
            self.X.append(item[-21:-1])
            self.y.append(item[-1])
        self.X = np.array(self.X)
        self.rawy = np.array(self.y)
        self.y = dense_to_one_hot(np.array(self.y, dtype=np.uint8))

        self.train_data = self.X[:int(len(self.X) * 0.8)]
        self.train_label = self.y[:int(len(self.y) * 0.8)]
        self.train_data_size = len(self.train_data)
        self.test_data = self.X[int(len(self.X) * 0.8):]
        self.test_label = self.y[int(len(self.y) * 0.8):]
        self.test_data_size = len(self.test_data)


    
def dense_to_one_hot(labels_dense, num_classes=23):
    """ Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

if __name__ == "__main__":
    file_list = combine_file_paths('dataset')
    for file in file_list:
        if file.endswith('.csv'):  # 只处理以 .csv 结尾的文件
            tmp_dataset = LoadDataset(file)
        print(f">>> For the dataset {tmp_dataset.dataset_name}.")
        print(f"> The shape of trainset is {tmp_dataset.train_data.shape}, \
              the shape of testset is {tmp_dataset.test_data.shape}, \
              the shape of testset's label is {tmp_dataset.test_label.shape}.")
        print(f"The 71st smaple's labels are {tmp_dataset.train_label[71]}")
    #     break
        # print(tmp_dataset.train_data[0])

    # tmp_dataset = LoadDataset('dataset/lucene-2.0.csv')
    # print(f">>> For the dataset {tmp_dataset.dataset_name}.")
    # print(f"> The shape of trainset is {tmp_dataset.train_data.shape}, \
    #       the shape of testset is {tmp_dataset.test_data.shape}, \
    #       the shape of label is {tmp_dataset.test_label.shape}.")
    # for i in range(75):
    #     print(tmp_dataset.train_label[i])
    