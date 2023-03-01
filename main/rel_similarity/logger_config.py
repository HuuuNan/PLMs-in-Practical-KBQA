#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
日志等级：使用范围

FATAL：致命错误
CRITICAL：特别糟糕的事情，如内存耗尽、磁盘空间为空，一般很少使用
ERROR：发生错误时，如IO操作失败或者连接问题
WARNING：发生很重要的事件，但是并不是错误时
INFO：处理请求或者状态变化等日常事务
DEBUG：调试过程中使用DEBUG等级，如算法中每个循环的中间状态
"""

import logging
import os


# 配置日志打印格式
def get_logger(log_file):
    log_dir = os.path.split(log_file)[0]
    # 如果日志文件目录未指定，则为当前文件目录。若指定但不存在，则创建目录
    if not log_dir.strip():
        log_dir = os.getcwd()
        print(log_dir)
    elif not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(log_file)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # logger中添加FileHandler，可以将日志写入到文件
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        # logger中添加StreamHandler，可以将日志输出到屏幕上
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # 设置输出格式
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # 添加Handler
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
