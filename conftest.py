'''
# -*- coding: utf-8 -*-
# @File    : conftest.py
# @Author  : Hua Guo
# @Time    : 2022/05/30
# @Disc    : conf for pytest
'''
import pytest


def pytest_configure():
  pytest.df = None
  pytest.X_train = None
  pytest.X_test = None
  pytest.y_train = None
  pytest.y_test = None