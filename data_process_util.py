# -*- coding:utf-8 -*-
# @author Fenglongyu
import json
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.utils import shuffle
from collections import defaultdict
from tokenizer import FullTokenizer
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# 单例模式
class ReadData(object):
    """
    (1)vocab.txt 和train.tsv, test.tsv放在同一个目录下
    (2)ReadData 类自动保存转为one-hot的train_data.json 和 label.json 在source目录下
    """
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path, save_pth='source/', ratio=0.1, length=100):

        self.dir_path = os.getcwd()
        self.train_save_pth = os.path.join(self.dir_path, save_pth + 'train_data.json')
        self.label_save_pth = os.path.join(self.dir_path, save_pth + 'label.json')
        self.test_save_pth = os.path.join(self.dir_path, save_pth + 'test_data.json')
        self.test_label_save_pth = os.path.join(self.dir_path, save_pth, 'test_label.json')
        self.train_path = os.path.join(path, 'train.tsv')
        self.length = length
        self.train_dataframe = pd.read_csv(self.train_path, delimiter='\t')
        self.vocab_path = os.path.join(path, 'vocab.txt')
        self.tokenizer = FullTokenizer(self.vocab_path)
        self.enc = LabelBinarizer()
        self.ratio = ratio

    def process(self):
        train = self.train_dataframe['Phrase'].to_numpy(dtype=str)
        label = self.train_dataframe['Sentiment'].to_numpy()

        train_data =[]
        for sentence in tqdm(train):
            tokens = self.tokenizer.tokenize(sentence)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            train_data.append(ids)

        # 补齐vector
        for sentence in train_data:
            if len(sentence)<self.length:
                while len(sentence)<self.length:
                    sentence.append(0)
            else:
                print("存在超过70")
        # enc = LabelBinarizer()
        # label = enc.fit_transform(label)
        mm = MinMaxScaler()
        train_data = mm.fit_transform(train_data)

        print("train_lenght", len(train_data))
        print("test_length", len(label))
        train_x, test_x, train_y, test_y = train_test_split(np.array(train_data), np.array(label), test_size=self.ratio,
                                                            random_state=5)
        train_x = train_x.tolist()
        test_x = test_x.tolist()
        train_y = train_y.tolist()
        test_y = test_y.tolist()
        print("length of train", len(train_x))
        print("length of label", len(train_y))

        for i in train_x:
            assert len(i) == 100

        try:
            f = open(self.train_save_pth, 'w')
            json.dump(train_x, f)
            f.close()
        except FileExistsError as F:
            print("保存路径不存在，异常为%s"%F)
        except Exception as e:
            print("异常为%s"%e)

        try:
            write = open(self.label_save_pth, 'w')
            json.dump(train_y, write)
            write.close()
        except FileExistsError as F:
            print("保存路径不存在，异常为%s" % F)
        except Exception as e:
            print("异常为%s"%e)

        try:
            f = open(self.test_save_pth, 'w')
            json.dump(test_x, f)
            f.close()
        except FileExistsError as F:
            print("保存路径不存在，异常为%s" % F)
        except Exception as e:
            print("异常为%s" % e)

        try:
            write = open(self.test_label_save_pth, 'w')
            json.dump(test_y, write)
            write.close()
        except FileExistsError as F:
            print("保存路径不存在，异常为%s" % F)
        except Exception as e:
            print("异常为%s" % e)


def shuffle_data(data):
    train, label = data
    return shuffle(train, label)


def batch_data(path,  train_or_test='train'):
    """make data generator in a batch_size"""
    # 数组对齐
    if train_or_test == 'train':
        file = open(os.path.join(path, 'train_data.json'), 'r', encoding='UTF-8')
        train = json.load(file)
        file.close()

        label_file = open(os.path.join(path, 'label.json'), 'r', encoding='UTF-8')
        label = json.load(label_file)
        label_file.close()
    elif train_or_test == 'test':
        file = open(os.path.join(path, 'test_data.json'), 'r')
        train = json.load(file)
        file.close()

        label_file = open(os.path.join(path, 'test_label.json'), 'r')
        label = json.load(label_file)
        label_file.close()

    return train, label

    # train, label = shuffle_data((np.array(train), np.array(label)))
    #
    # data_len = len(train)
    # steps_per_epoch = data_len // batch_size
    # for i in range(steps_per_epoch):
    #     yield (train[i*batch_size:i * batch_size + batch_size], label[i*batch_size:i * batch_size + batch_size])


def main():
    project_pth = os.path.join(os.getcwd(),'source')
    path = r"D:\celeba4\project_env\base_nlp\data\sentiment-analysis-on-movie-reviews"
    if not (os.path.exists(os.path.join(project_pth,'train_data.json')) and os.path.exists(os.path.join(project_pth, 'label.json'))):
        print("正在读取数据，并且处理。。。")
        readData = ReadData(path)
        readData.process()
    else:
        print("使用已处理好的处理，直接开始分batches")


if __name__ =="__main__":
    main()
