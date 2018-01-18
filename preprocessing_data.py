import csv
import numpy as np
from lib import *
from sklearn import preprocessing
# data X for training/test : [[X1],[X2],...,[Xn]]
# data y for training/test : [y1,y2,...,yn]

debug = False
p = Print(debug)


class Preprocessing:
    def __init__(self):
        self.input = []
        pass

    def preprocessing(self, filename, col=1):
        p.print('==== preprocessing oil gland feature ====')
        f = open(filename, 'r')
        if col == 1:
            while True:
                row = f.readline()
                if row is '':
                    break
                self.input.append(float(row))
        p.print(self.input)
        self.input = preprocessing.normalize(self.input)[0]
        # self.input = np.array(self.input)
        self.input = self.input[:,np.newaxis]
        p.print(self.input)
        # return self.input

    def get_input(self):
        return self.input

    def get_output(self):
        return list(range(1,len(self.input)+1,1))




if __name__ == '__main__':
    pass
