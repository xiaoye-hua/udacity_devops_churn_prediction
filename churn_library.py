# library doc string

# import libraries
import os
import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


logging.basicConfig(
    # filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth, debug=False) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    if debug:
        df = pd.read_csv(r"./data/bank_data.csv", nrows=100)
    else:
        df = pd.read_csv(r"./data/bank_data.csv")
    return df


def perform_eda(df: pd.DataFrame, img_dir='images/') -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    plt.savefig(os.path.join(img_dir, 'marital_status_bar.png'))


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df: pd.DataFrame, response=None) -> Tuple[pd.DataFrame]:
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    y = df['Churn']
    X= df.drop('Churn', axis=1)[['Total_Relationship_Count', 'Total_Trans_Amt']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test, debug=False) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)


    if debug:
        param_grid = {
            'n_estimators': [200],
            'max_features': ['auto'],
            'max_depth': [4,],
            'criterion': ['gini']
        }
    else:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    logging.info("Grid Searching")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=True)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

def main(debug=False):
    '''
    main function of this script
    '''
    # config
    data_pth = r"./data/bank_data.csv"
    logging.info('Reading data...')
    df = import_data(pth=data_pth,debug=debug)
    logging.info('Performing EDA...')
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    logging.info("Model training...")
    train_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, debug=debug)


if __name__ == '__main__':
    debug = True
    logging.info(f'Debug mode:{debug}')
    main(debug=debug)