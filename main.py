# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import math
from scipy.stats import kurtosis
from scipy import stats, special
from scipy.stats import skew
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
import warnings
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models
from keras import layers

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 1000)


def plot_hist_boxplot(column):
    fig, [ax1, ax2]=plt.subplots(1,2,figsize=(12,5))
    sns.distplot( train[train[column].notnull()][column],ax=ax1)
    sns.boxplot(y=train[train[column].notnull()][column],ax=ax2)
    print("Slewness : ", skew(train[train[column].notnull()][column]))
    print("kurtosis : ", kurtosis(train[train[column].notnull()][column]))
    plt.show()
def plot_count_boxplot(column):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
    sns.countplot(train[train[column].notnull()][column],ax=ax1)
    sns.boxplot(y=train[train[column].notnull()][column],ax=ax2)
    print("Slewness : ", skew(train[train[column].notnull()][column]))
    print("kurtosis : ", kurtosis(train[train[column].notnull()][column]))
    plt.show()

def boxplot_violinplot(col1, col2):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x=col1,y=col2,data=train,pallete='Set3',ax=ax1)
    sns.violinplot(x=col1,y=col2,data=train,palette='Set3',ax=ax2)
    plt.show()


def print_hi(name):
    global train
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # read training dataset
    df = pd.read_csv('Data/cs-training.csv')
    print(df.shape)
    df.info()
# what percentage of data is missng in the feature
    print( round(df.isnull().sum(axis=0)/len(df),2)*100)
    print( df.head() )
    df['Unnamed: 0'].nunique()/len(df)
    df.rename( columns = {'Unamed: 0' : 'CustomerID'}, inplace=True)

    # Target Variable
    print(df['SeriousDlqin2yrs'].unique())
    print()
    print('{}% of the borrowers falling in the serious delinquency '.format((df['SeriousDlqin2yrs'].sum()/len(df))*100))

    fig, axes = plt.subplots(1,2,figsize=(12,6))
    df['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=axes[0])
    axes[0].set_title('SeriousDlqin2yrs')
    axes[1].set_title('SeriousDlqin2yrs')
    sns.countplot(data=df,ax=axes[1])
    plt.show()

    print( df['SeriousDlqin2yrs'].value_counts() )
    print ( df.describe() )

# split dataset into train and test
    data = df.drop( columns = ['SeriousDlqin2yrs'], axis=1)
    y = df['SeriousDlqin2yrs']
    from sklearn.model_selection import train_test_split
    df_test, df_train, y_test, y_train = train_test_split(data, y, test_size=0.8, random_state=42, stratify=y)
    print( df_test.shape, df_train.shape )

    print('Event rate in the training dataset : ', np.mean(y_train))
    print()
    print('Event rate in the test dataset : ', np.mean(y_test))
    print()
    print('Event rate in the training dataset : ', np.mean(y))
    print()

    train = pd.concat([df_train, y_train],axis=1)
    print( train.shape )

    test = pd.concat([df_train, y_train], axis=1)
    print(train.shape)

    plot_hist_boxplot('RevolvingUtilizationOfUnsecuredLines')
    plot_hist_boxplot('age')
    plot_hist_boxplot('DebtRatio')
    plot_hist_boxplot('MonthlyIncome')
    plot_hist_boxplot('NumberOfOpenCreditLinesAndLoans')
    plot_hist_boxplot('NumberRealEstateLoansOrLines')
    plot_count_boxplot('NumberOfDependents')
    plot_hist_boxplot('NumberOfTime30-59DaysPastDueNotWorse')
    plot_hist_boxplot('NumberOfTime60-89DaysPastDueNotWorse')
    plot_hist_boxplot('NumberOfTimes90DaysLate')

    cols_for_stats = ['RevolvingUtilizationOfUnsecuredLines',
                      'age',
                      'NumberOfTime30-59DaysPastDueNotWorse',
                      'DebtRatio',
                      'MonthlyIncome',
                      'NumberOfOpenCreditLinesAndLoans',
                      'NumberOfTimes90DaysLate',
                      'NumberOfRealEstateLoansOrLines',
                      'NumberOfTimes60-89DaysPastDueNotWorse',
                      'NumberOfDependents']

    skewness = []
    kurt = []
    for column in cols_for_stats:
        skewness.append(skew(train[train[column].notnull()][column]))
        kurt.append(kurtosis(train[train[column].notnull()][column]))

    stats = pd.DataFrame({'skewness':skewness, 'kurtosis':kurt}, index=[col for col in cols_for_stats])
    stats.sort_values(by='skewness', ascending=False)
    print(stats)

    plt.figure(figsize=(10,6))
    sns.heatmap(train.corr(), annot=True, cmap=plt.cm.CMRmap_r)
    plt.show()

    boxplot_violinplot('SeriousDlqin2yrs', 'age')
    boxplot_violinplot('SeriousDlqin2yrs', 'MonthlyIncome')
    boxplot_violinplot('SeriousDlqin2yrs', 'DebtRatio')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfOpenCreditLinesAndLoans')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfRealEstateLoansOrLines')
    boxplot_violinplot('SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfDependents')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfTime30-59DaysPastDueNotWorse')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfTime60-89DaysPastDueNotWorse')
    boxplot_violinplot('SeriousDlqin2yrs', 'NumberOfTimes90DaysLate')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
