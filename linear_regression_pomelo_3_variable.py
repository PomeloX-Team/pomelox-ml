import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score#, r2_core
from preprocessing_data import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



def main():
    pre = Preprocessing()
    pre.preprocessing('Pomelo_features_A_1_3')
    pomelo_X_train = pre.get_input()
    pomelo_y_train = pre.get_output()

  

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

    
    # plt.scatter(pomelo_X_train, pomelo_y_train,  color='yellow', s=30, marker='o', label="training points")
    # plt.plot(pomelo_X_test, pomelo_y_pred, color='red', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()

if __name__ == '__main__':
    main()
   