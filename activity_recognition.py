import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt
import math
import scipy
from scipy.stats import skew, kurtosis
from statistics import mean, variance, stdev
import statsmodels.tsa.stattools as ts  # Statistical tools for time series analysis
import multiprocessing
import itertools
#%matplotlib inline

USER_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ACCEL_IDS = [i for i in range(1, 301)]
np.random.seed(0)


def get_accel(user_id, accel_id, folder):
    filename = '%s/%s/%s.csv' % (folder, 'user' + str(user_id), 'accelerometer_' + str(accel_id))
    COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
    df = pd.read_csv(filename, header=None, names=COLUMNS)
    accel = [[df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]] for i in range(len(df))][:142]
    return accel


def get_accels(user_id, folder):
    if folder == 'train':
        for accel_id in range(1, 201):
            yield get_accel(user_id, accel_id, folder)
    elif folder == 'test':
        for accel_id in range(201, 301):
            yield get_accel(user_id, accel_id, folder)


def get_random_accels(user_id, folder):
    users = [i for i in range(1, 11)]
    users.remove(user_id)
    if folder == 'train':
        for user_id in users:
            accel_ids = np.random.choice(np.arange(1, 201), 23, replace=False)
            accel_ids = list(accel_ids)
            for accel_id in accel_ids:
                yield get_accel(user_id, accel_id, folder)
    elif folder == 'test':
        for user_id in users:
            accel_ids = np.random.choice(np.arange(201, 301), 12, replace=False)
            accel_ids = list(accel_ids)
            for accel_id in accel_ids:
                yield get_accel(user_id, accel_id, folder)

        # print(len(list(get_random_accels(1, 'train'))))      23*9=207 accels randomly sampled from the other users
        # print(len(list(get_random_accels(1, 'test'))))      12*9=108 accels randomly sampled from the other users

    def plot_axis(ax, x, y, title):
        ax.plot(x, y)
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)

    def plot_accel(user_id, accel_id, folder):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
        filename = '%s/%s/%s.csv' % (folder, 'user' + str(user_id), 'accelerometer_' + str(accel_id))
        COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
        df = pd.read_csv(filename, header=None, names=COLUMNS)
        plot_axis(ax0, df['timestamp'], df['xAxis'], 'x Axis')
        plot_axis(ax1, df['timestamp'], df['yAxis'], 'y Axis')
        plot_axis(ax2, df['timestamp'], df['zAxis'], 'z Axis')
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    def magnitude_acc(accel):
        magn = []
        for i in range(len(accel)):
            m2 = accel[i][1] ** 2 + accel[i][2] ** 2 + accel[i][3] ** 2
            magn.append(math.sqrt(m2))
        return magn


# print(magnitude_acc(get_accel(user_id, accel_id, folder)))

def windows(df, size=30):
    start = 0
    while (start + size) < len(df):
        yield start, start + size
        start += (size / 2)


def jitter(axis, start, end):
    j = float(0)
    for k in range(start, min(end, len(axis))):
        if start > 0:
            j = j + abs(axis[k] - axis[k - 1])
    return j / float(end - start)


def mean_crossing_rate(axis, start, end):
    cr = 0
    m = mean(axis)
    for i in range(start, min(end, len(axis))):
        if start > 0:
            p = axis[i - 1] > m
            c = axis[i] > m
            if p != c:
                cr += 1
    return float(cr) / (end - start - 1)


def window_summary(axis, start, end):
    start = int(start)
    end = int(end)
    # acf = ts.acf(np.array(axis[start:end]))  #auto correlation
    # acv = ts.acovf(np.array(axis[start:end]))  #auto covariance
    return [
        # jitter(axis, start, end),
        mean_crossing_rate(axis, start, end),
        mean(axis[start:end]),
        stdev(axis[start:end]),
        variance(axis[start:end]),
        min(axis[start:end]),
        max(axis[start:end]),
        # acf.mean(), # mean auto correlation
        # acf.std(), # standard deviation auto correlation
        # acv.mean(), # mean auto covariance
        # acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end])
    ]


def build_features(accel):
    timestamps = []
    xAxis = []
    yAxis = []
    zAxis = []
    for i in range(len(accel)):
        timestamps.append(accel[i][0])
        xAxis.append(accel[i][1])
        yAxis.append(accel[i][2])
        zAxis.append(accel[i][3])
    features = []
    for (start, end) in windows(timestamps):
        start = int(start)
        end = int(end)
        features += window_summary(xAxis, start, end)
        features += window_summary(yAxis, start, end)
        features += window_summary(zAxis, start, end)
    return features


def get_data(user_id):
    list1 = list(get_accels(user_id, 'train'))  # first 200 observations of the training set
    list2 = list(get_random_accels(user_id, 'train'))[:-7]  # second 200 observations of the training set
    list12 = list1 + list2  # all 400 observations of the training set
    list1 = [build_features(accel) for accel in list12]  # features for each observation of the training set

    list3 = list(get_accels(user_id, 'test'))  # first 100 observations of the testing set
    list4 = list(get_random_accels(user_id, 'test'))[:-8]  # second 100 observations of the testing set
    list34 = list3 + list4  # all 200 observations of the testing set
    list2 = [build_features(accel) for accel in list34]  # features for each observation of the testing set
    return list1, list2


def run_model(user_id, Model):
    trainX, testX = get_data(user_id)
    trainY = [1] * 200 + [0] * 200  # labels with 1 for the correct user and 0 for the incorrect user in training
    testY = [1] * 100 + [0] * 100  # labels with 1 for the correct user and 0 for the incorrect user in testing
    model = Model.fit(trainX, trainY)
    predictions = model.predict_proba(testX)[:, 1]  # probability estimates of the positive class
    return testY, predictions


if __name__ == '__main__':
    # Model = GradientBoostingClassifier(n_estimators=500,learning_rate=1,max_depth=5,random_state=0)
    Model = RandomForestClassifier(n_estimators=500, random_state=0)
    # results=[]
    # for i in range(1,11):
    #    results.extend(list(run_model(i, Model)))
    for i in range(1, 11):
        testY, predictions = run_model(i, Model)
        fpr, tpr, thresholds = roc_curve(testY, predictions)
        #print('predictions=', predictions)
        roc_auc = auc(fpr, tpr)
        print(roc_auc*100, '%')