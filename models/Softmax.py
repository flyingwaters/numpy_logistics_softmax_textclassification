# -*- coding:utf-8 -*-
# @author Fenglongyu

import sys
import numpy as np
sys.path.append('../')
from data_process_util import batch_data



class Softmax(object):
    """model logistic/regression """
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=3000,
              batch_size=64, verbose=True):

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        print(y.shape)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        # enc = LabelBinarizer()
        # y = enc.fit_transform(y)
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx, :]
            y_batch = y[idx]


            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def loss(self, X_batch, y_batch, reg):
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_train = X_batch.shape[0]
        WX = np.matmul(X_batch, self.W)
        out_put = np.exp(WX - np.max(WX)) / (np.sum(np.exp(WX-np.max(WX)), axis=1, keepdims=True)+0.00001)
        loss = 1 / num_train * np.sum(-np.log(out_put[range(num_train), list(y_batch)])) + 0.5 * reg * np.sum(
            self.W * self.W)
        out_put[range(num_train), list(y_batch)] -= 1
        out_put = np.matmul(X_batch.T, out_put)
        dW = out_put / num_train + reg * self.W

        return loss, dW

    def predict(self, X):
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred


if __name__ == "__main__":
    path = r"D:\celeba4\project_env\base_nlp\source"
    model = Softmax()
    train, label = batch_data(path)
    model.train(X=np.array(train), y=np.array(label), num_iters=80000, batch_size=64)
