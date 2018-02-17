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

# yi =  b1 xi,1 +  b2 xi,2 +  b3 xi,3 +  ui

def main():
    pre = Preprocessing()
    pre.preprocessing('Pomelo_features_A_1_3')

    pomelo_X_train = pre.get_input()
    pomelo_y_train = pre.get_output()
    pomelo_plot_data = pre.get_data_plot()
  

    pre = Preprocessing()
    pre.preprocessing('Pomelo_features_A_4')  
    
    pomelo_X_test = pre.get_input()
    pomelo_y_test = pre.get_output()

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # pomelo_y_train,
    print(pomelo_y_test)
    print(pomelo_y_pred)
    c = np.array(pomelo_y_test)    
    ct = 1
    c_index = 0
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
        ax.scatter(data[0]*100,data[1]*100,data[2]*100, c=color, marker='^')
    
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
        ax.scatter(data[0]*100,data[1]*100,data[2]*100, c=color, marker='o')
    # CIRCLE = TRAIN
    # , pomelo_y_pred
    # plt.plot(pomelo_X_test[0],pomelo_X_test[1],pomelo_X_test[2], color='red', cmap=plt.hot())

    # plt.xticks(())
    # plt.yticks(())

    plt.show()

if __name__ == '__main__':
    main()
   