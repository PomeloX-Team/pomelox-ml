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
                
            self.input = preprocessing.normalize(self.input)[0]
            self.input = self.input[:,np.newaxis]
        elif col == 3:
            while True:
                row = f.readline()
                if row is '':
                    break
                row = row.split(',')
                self.input.append([float(row[0]),float(row[1]),float(row[2])])
            self.input = np.array(self.input)
            self.input = preprocessing.normalize(self.input)
        p.print(self.input)
        # return self.input


    def get_input(self):
        return self.input

    def get_output(self):
        return list(range(1,len(self.input)+1,1))




if __name__ == '__main__':
    # p.change_mode(True)
    pp = Preprocessing()
    pp.preprocessing('histogram_A1',3)
    print(pp.get_input())
    pass
