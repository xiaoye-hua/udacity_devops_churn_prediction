"""
# -*- coding: utf-8 -*-
# @File    : churn_script_loggin_and_tests.py
# @Author  : Hua Guo
# @Time    : 2022/05/30
# @Disc    : test for churn_library.py
"""
import os
import logging
import pytest
# import churn_library_solution as cls
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv", debug=True)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    # request.config.cache.set("df", df)
    # request.config.cache.set()
    pytest.df = df


def test_eda():
    '''
    test perform eda function
    '''
    df = pytest.df
    assert df is not None
    perform_eda(df)


def test_encoder_helper(  # encoder_helper
):
    '''
    test encoder helper
    '''
    df = pytest.df
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    target_cols = [col + "_Churn" for col in cat_columns]
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df = encoder_helper(df=df, category_lst=cat_columns, response=target_cols)
    assert target_cols[3] in df.columns


def test_perform_feature_engineering(  # perform_feature_engineering
):
    '''
    test perform_feature_engineering
    '''
    df = pytest.df
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test = X_train, X_test, y_train, y_test


def test_train_models(  # train_models
):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = pytest.X_train, pytest.X_test, pytest.y_train, pytest.y_test
    assert X_test is not None
    assert y_train is not None
    train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        debug=True)
