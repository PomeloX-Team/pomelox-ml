'''
    File name: linear_regression.py
    Author: PomeloX
    Date created: 1/18/2018
    Date last modified: 2/15/2018
    Python Version: 3.6.1
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score
from preprocessing_data import *
import random
import operator
# yi =  b1 xi,1 +  b2 xi,2 +  b3 xi,3 +  ui

def main():
    number_of_gland_data = []
    pre = Preprocessing()
    pre.preprocessing(['pomelo_features_number of gland_A1.csv',
                        'pomelo_features_number of gland_A2.csv',
                        'pomelo_features_number of gland_A3.csv',
                        'pomelo_features_number of gland_A4.csv'
    ])
    NOG_A = pre.get_input()
    # NOG_a1 = pre.get_input()
    # pre = Preprocessing()    
    # pre.preprocessing()
    # NOG_a2 = pre.get_input() 
    # pre = Preprocessing()
    # pre.preprocessing('pomelo_features_number of gland_A3.csv')
    # NOG_a3 = pre.get_input() 
    # pre = Preprocessing()
    # pre.preprocessing('pomelo_features_number of gland_A4.csv')
    # NOG_a4 = pre.get_input() 

    radius_data = []
    pre = Preprocessing()
    pre.preprocessing(['pomelo_features_radius_A1.csv',
                        'pomelo_features_radius_A2.csv',
                        'pomelo_features_radius_A3.csv',
                        'pomelo_features_radius_A4.csv'                    
    ])
    radius_A = pre.get_input()
    # radius_a1 = pre.get_input()
    # pre = Preprocessing()    
    # pre.preprocessing('pomelo_features_radius_A2.csv')
    # radius_a2 = pre.get_input() 
    # pre = Preprocessing()
    # pre.preprocessing('pomelo_features_radius_A3.csv')
    # radius_a3 = pre.get_input() 
    # pre = Preprocessing()
    # pre.preprocessing('pomelo_features_radius_A4.csv')
    # radius_a4 = pre.get_input() 
    
    y = pre.get_output()

    data = []

    # for (i,j,k) in zip(NOG_A,radius_A,y):
    #     data.append([[i[0],j[0]],k])

    for (i,j,k) in zip(NOG_A,radius_A,y):
        data.append([[i[0]],k])
    train = random.sample(data,int(len(data)*0.8))

    pomelo_X_train = []
    pomelo_y_train = []
    pomelo_X_test = []
    pomelo_y_test = []

    for i in train:
        data.remove(i)
        pomelo_X_train.append(i[0])
        pomelo_y_train.append(i[1])
    
    for i in data:
        pomelo_X_test.append(i[0])
        pomelo_y_test.append(i[1])
    
    # for (i,j) in zip
    # pre = Preprocessing()
    # pre.preprocessing('Pomelo_features_A_4')  
    
    # pomelo_X_test = pre.get_input()
    # pomelo_y_test = pre.get_output()
    print(pomelo_X_train,pomelo_y_train)
    regr = linear_model.LinearRegression()
    regr.fit(pomelo_X_train, pomelo_y_train)

    # Make predictions using the testing set
    pomelo_y_pred = regr.predict(pomelo_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)

    # The mean squared error
    mse = mean_squared_error(pomelo_y_test, pomelo_y_pred)
    print("Mean squared error: "+str(mse))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: '+str(explained_variance_score(pomelo_y_test, pomelo_y_pred)))

    # fig = plt.figure()
    # pomelo_y_train,
    print(pomelo_X_test)
    print(pomelo_y_test)
    print(pomelo_y_pred)
    c = np.array(pomelo_y_test)    
    ct = 1
  
    
    for data,c in zip(pomelo_X_train,pomelo_y_train):
        if c <= 7:
            color = 'brown'
        elif c <= 14:
            color = 'red'
        elif c <= 21:
            color = 'orange'
        elif c <= 28:
            color = 'yellow'
        elif c <= 35:
            color = 'lime'
        elif c <= 42:
            color = 'green'
        elif c <= 49:
            color = 'navy'
        else:
            color = 'black'
        plt.plot(data[0],c)

    for data,c in zip(pomelo_X_test,pomelo_y_pred):
        if c <= 7:
            color = 'brown'
        elif c <= 14:
            color = 'red'
        elif c <= 21:
            color = 'orange'
        elif c <= 28:
            color = 'yellow'
        elif c <= 35:
            color = 'lime'
        elif c <= 42:
            color = 'green'
        elif c <= 49:
            color = 'navy'
        else:
            color = 'black'
        plt.plot(data[0],c)

    
    # CIRCLE = TRAIN
    # , pomelo_y_pred
    # 

    # plt.xticks(())
    # plt.yticks(())

    plt.show()

if __name__ == '__main__':
    main()
   