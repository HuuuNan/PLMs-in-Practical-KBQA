#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
import time
import os
import functools
import socket
import numpy as np
import subprocess
import io
import config.global_var as gl

from utils.logger_config import get_logger

import os


logger = get_logger(gl.KBQA_LOG_PATH)

# 获取当前时间（秒级）
def get_now_time():
    times = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return times


# 获取当前时间戳（秒级）
def get_now_timestamp():
    timestamp = int(time.time())
    return timestamp


# 将时间转化为时间戳
def time_to_timestamp(times):
    time_array = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(time_array))
    return timestamp


# 将时间戳转化为时间
def timestamp_to_time(timestamp):
    time_local = time.localtime(timestamp)
    times = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return times


# 秒到时分秒的转换
def sec_to_time(sec):
    ''' Convert seconds to '#D days#, HH:MM:SS.FFF' '''
    if hasattr(sec, '__len__'):
        return [sec_to_time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    pattern = r'%02d:%02d:%02d'
    return pattern % (h, m, s)


# 判断文件夹是否存在，不存在就新建
def check_folder_exist(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


# 装饰器，记录函数运行时间
def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = fn(*args, **kw)
        end_time = time.time()
        print('耗时： %s s' % (end_time - start_time))
        return result
    return wrapper


# 将字典的key转换为字符串类型
def transform_dict_key_int_to_str(dic):
    dic = {str(k): v for k, v in dic.items()}
    return dic


# 将字典的key转换为int类型
def transform_dict_key_str_to_int(dic):
    dic = {int(k): v for k, v in dic.items()}
    return dic


# 将字典的key、value对调
def exchange_dict_key_value(dic):
    dic = {v: k for k, v in dic.items()}
    return dic


# 发布模型到docker中
def model_to_docker(serving_model_path, model_id, port):
    cmd = 'docker run -d --restart=always --name tf-{} -p {}:8500 -v {}:/models/{}: -e MODEL_NAME={} ' \
          '-t tensorflow/serving &'.format(str(model_id), str(port), serving_model_path, str(model_id), str(model_id))
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
    stream_stdout = io.TextIOWrapper(proc.stdout, encoding='utf-8')
    stream_stderr = io.TextIOWrapper(proc.stderr, encoding='utf-8')
    cmd_out = str(stream_stdout.read())
    cmd_err = str(stream_stderr.read())

    logger.debug('cmd_out:{}'.format(str(cmd_out)))
    logger.debug('cmd_err:{}'.format(str(cmd_err)))

    if not cmd_out and cmd_err:
        docker_info = 'error'
    elif not cmd_err and cmd_out:
        docker_info = 'info'
    else:
        docker_info = None
    return cmd, docker_info, port


# 删除docker模型
# cmd kill docker，rm docker
def delete_docker_model(model_id):
    docker_name = 'tf-{}'.format(model_id)
    os.system('docker rm -f {}'.format(docker_name))


# 获取空闲端口
def get_available_port(ip, min_int, max_int):
    port = 99999
    flag = True
    while flag:
        port = np.random.randint(min_int, max_int)
        flag = is_port_used(ip, port)
    return port


# 端口占用查询
def is_port_used(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        # 利用shutdown()函数使socket双向数据传输变为单向数据传输。shutdown()需要一个单独的参数，
        # 该参数表示了如何关闭socket。具体为：0表示禁止将来读；1表示禁止将来写；2表示禁止将来读和写。
        return True
    except:
        return False