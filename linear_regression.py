from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.model_selection import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
"""
Based on given input data we have 9 columns, but not all of them are related to the price
Therefor, data is diveded in independent and dependent variable

Independent are: 3 - 7
Dependent is 9

column: 3 -> complexAge
column: 4 -> totalRooms
column: 5 -> totalBedrooms
column: 6 -> complexInhabitants
column: 7 -> apartmentsNr
column: 9 -> medianComplexValue
"""

headers = ["Unknown_1", "Unknown_2", "complexAge", "totalRooms", "totalBedrooms",
           "complexInhabitants", "apartmentsNr", "Unknown_8", "medianComplexValue"]


def read_data(name: str):
    """

    :param name: file path
    :return: Data frame
    """

    # read file and assign header names
    data = pd.read_csv(name, sep=",", names=headers)
    # check if there is any null value
    if data.isnull().sum().sum() == 0:
        return data
    else:
        raise Exception("Null values were found")


def compute_regression(data, dependent_var_position: int):
    """

    :param data: data frame from input file
    :param dependent_var_position: position of dependent variable in data frame
    :rtype: tuple
    :return: LinearRegression object and linear regression accuracy
    """
    features = filtering_by_corr(data, 0)
    independent_vars = data[features]
    dependent_var = data.iloc[:, dependent_var_position].values

    # Define testing and train data
    x_train, x_test, y_train, y_test = train_test_split(independent_vars, dependent_var, test_size=0.1, random_state=10)

    linear_regression = LinearRegression()

    linear_regression.fit(x_train, y_train)
    # test regresion with 20% of data
    prediction = linear_regression.predict(x_test)
    # compute  accuracy
    plot_graph((y_test, prediction))

    statistic_test = {
        'variance_score': round(explained_variance_score(y_test, prediction), 2),
        'rmse': round(sqrt(mean_squared_error(y_test, prediction)), 2),
        'r2_score': round(r2_score(y_test, prediction), 2)
    }

    return linear_regression, statistic_test


def predict(linear_regression, input_data):
    return linear_regression.predict([input_data])[0]


def heatmap(data):
    corr = data.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)
    plt.show()


def plot_graph(data: tuple):
    actual_value = data[0]
    predicted_value = data[1]
    res = pd.DataFrame({'Predicted': predicted_value, 'Actual': actual_value})
    res = res.reset_index()
    res = res.drop(['index'], axis=1)
    plt.plot(res[:30])
    plt.legend(['Actual', 'Predicted'])
    plt.show()


def pre_process_data(data):
    """
    outlier using percentage method
    pre process data, remove data that looks strange and not realistic
    filter data which are less than 1% and greater than 95%
    """
    min_thresold, max_thresold = data.medianComplexValue.quantile([0.001, 0.950])
    return data[(data.medianComplexValue < max_thresold) & (data.medianComplexValue > min_thresold)]


def standard_deviation(data, order):
    plt.hist(data.medianComplexValue, bins=20, rwidth=0.8, density=True)
    plt.xlabel('Median Complex Value')
    plt.ylabel('Count')
    rng = np.arange(0, data.medianComplexValue.max(), 100)
    plt.plot(rng, norm.pdf(rng, data.medianComplexValue.mean(), data.medianComplexValue.std()))
    plt.show()
    data['z_score'] = 0
    data.loc[:, ('z_score')] = (data.medianComplexValue - data.medianComplexValue.mean()) / data.medianComplexValue.std()
    return data[(data.z_score < order) & (data.z_score > -order)].drop('z_score', axis=1)


def normalization(data):
    x = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_frame = pd.DataFrame(x_scaled)
    data_frame.columns = headers
    return data_frame


def filtering_by_corr(data, limit):
    dt = abs(data.corr()['medianComplexValue'][abs(data.corr()['medianComplexValue']) > limit].drop(
        "medianComplexValue")).index.tolist()
    return dt
