import csv
import numpy as np
from lib import *
from sklearn import preprocessing
import constant as CONST
# data X for training/test : [[X1=x11,x12,..,x1k],[X2],...,[Xn]]
# data y for training/test : [y1,y2,...,yn]

debug = False
p = Print(debug)


class Preprocessing:
    def __init__(self):
        self.input = []
        self.output = []
        self.data_plot = [[],[],[]] 
        pass

    def preprocessing(self, filenames):
        for filename in filenames:
            p.print('==== preprocessing oil gland feature ====')
            print(filename)
            f = open(CONST.CSV_PATH+filename, 'r')
            row = f.readline()
            
            while True:
                row = f.readline()
                if row is '':
                    break
                row = row.split(',')
                self.input.append([float(row[0])]),#float(row[1])])
    
                self.data_plot[0].append(float(row[0]))
                # self.data_plot[1].append(float(row[1]))
    
                # self.output.append(int(row[2]))
                self.output.append(int(row[1]))

        self.input = np.array(self.input)
            # self.input = preprocessing.normalize(self.input,norm='max')
        p.print(self.input)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_data_plot(self):
        return self.data_plot


if __name__ == '__main__':
    # p.change_mode(True)
    pp = Preprocessing()
    pp.preprocessing('Pomelo_features_B')
    print(pp.get_input())
    print(pp.get_output())
    pass
