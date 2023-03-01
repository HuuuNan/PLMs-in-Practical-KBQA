#!/usr/bin/python3
# -*- coding: utf-8 -*-


class TrainParametersException(Exception):
    '''训练参数异常类'''
    def __init__(self, message='训练参数异常'):
        self.message = message


class DataProcessException(Exception):
    '''数据处理异常类'''
    def __init__(self, message='数据处理异常'):
        self.message = message