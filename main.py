# -*- coding:utf-8 -*-
# @author Fenglongyu

from models.Softmax import Softmax
from data_process_util import batch_data
import numpy as np


def main():
    """
    :return:
    """
    # Train
    path = r"D:\celeba4\project_env\base_nlp\source"
    model = Softmax()
    train, label = batch_data(path)
    model.train(X=np.array(train), y=np.array(label), num_iters=80000, batch_size=64)


if __name__ == '__main__':
    main()
