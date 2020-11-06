import numpy as np
import pandas as pd


class Dataset(object):

    def __init__(self, filename, column_names=None, drop_missing=False):

        self.dataset = pd.read_csv(filepath_or_buffer=filename,
                                   names=column_names, header=None, sep=",")
        self.drop_headred = self.dataset.drop(labels=0)

        if drop_missing:
            self.drop_missing_values()

    def drop_missing_values(self, missing_value=" ?"):
        print('Missing values')
        for i, j in zip(self.dataset.columns, (self.dataset.values.astype(str) == ' ?').sum(axis=0)):
            if j > 0:
                print(str(i) + ': ' + str(j) + ' records')

        self.dataset.replace({missing_value: np.nan}, inplace=True)
        self.dataset.dropna(axis=0, how="any", inplace=True)

        print('Dataset columns and their types after dropping missing values')
        print(self.dataset.info())

    @property
    def feature(self):
        return self.drop_headred.drop(columns=self.dataset.columns[-1])

    @property
    def onehot_encoded_output(self):
        return pd.get_dummies(self.drop_headred[self.drop_headred.columns[-1]])

    @property
    def label_encoded_output(self):
        return self.drop_headred[self.drop_headred.columns[-1]].astype('category').cat.codes

