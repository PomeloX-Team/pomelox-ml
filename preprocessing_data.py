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
        self.output = []
        pass

    def preprocessing(self, filename):
        p.print('==== preprocessing oil gland feature ====')
        f = open(filename, 'r')
        row = f.readline()
        
        while True:
            row = f.readline()
            if row is '':
                break
            row = row.split(',')
            self.input.append([float(row[0]),float(row[1]),float(row[2])])
            self.output.append(int(row[4]))
        self.input = np.array(self.input)
        self.input = preprocessing.normalize(self.input)
        p.print(self.input)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output




if __name__ == '__main__':
    # p.change_mode(True)
    pp = Preprocessing()
    pp.preprocessing('Pomelo_features_B')
    print(pp.get_input())
    print(pp.get_output())
    pass
