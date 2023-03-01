import os
import re
import time

import pymongo
import pymysql
import datetime
import config.global_var as gl

from utils.logger_config import get_logger
import requests
import json

# log文件
logger = get_logger(gl.KBQA_LOG_PATH)

# 连接数据库
myclient = pymongo.MongoClient(gl.CLIENT_NAME)
mydb = myclient[gl.KG_DB_NAME]


# 获取前1天或N天的日期，beforeOfDay=1：前1天；beforeOfDay=N：前N天
def getdate(beforeOfDay):
    today = datetime.datetime.now()
    # 计算偏移量
    offset = datetime.timedelta(days=-beforeOfDay)
    # 获取想要的日期的时间
    re_date = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
    return re_date


# 获取前一周的所有日期(weeks=1)，获取前N周的所有日期(weeks=N)
def getBeforeWeekDays(weeks=1):
    # 0,1,2,3,4,5,6,分别对应周一到周日
    week = datetime.datetime.now().weekday()
    start = 7 * weeks + week
    end = week
    for index in range(start, end, -1):
        print(index)
        day = getdate(index)
        print(day)


def convert_time(date_time):
    ts = int(time.mktime(time.strptime(date_time, "%Y-%m-%d %H:%M:%S")))
    # print(ts)
    return ts


def convert_stamp_time(time_stamp):
    if len(str(int(time_stamp))) == 10:
        time_stamp = time_stamp
    else:
        time_stamp = time_stamp / 1000
    timeArray = time.localtime(time_stamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    print(otherStyleTime)  # 2013--10--10 23:40:00
    return otherStyleTime


def get_file_upate_time(fileName):
    return os.path.getmtime(fileName) * 1000


def write_file(file_path, content):
    r1 = "[^\u4e00-\u9fa5^a-z^A-Z^0-9]"
    with open(file_path, "a+", encoding="utf8")as f:
        data = f.readlines()
    text = re.sub(r1, "", content).lower()
    content = text + "\n"
    if content not in data:
        data.append(content)
        logger.debug(content)
        f.write(content)


def write_simfile(sim_file, name, type):
    '956型驱逐舰	956型驱逐舰@@@@@驱逐舰'
    r1 = "[^\u4e00-\u9fa5^a-z^A-Z^0-9]"
    text = re.sub(r1, "", name).lower()
    content = name + "\t" + text + "@@@@@" + type + "\n"
    with open(sim_file, "a+", encoding="utf8")as f:
        data = f.readlines()
    if content not in data:
        data.append(content)
        logger.debug(content)
        f.write(content)


def write_names(file, id, name):
    'b23caa5aefef8ec9f19c474af6823b55@@@巡洋舰@@@“开济”号巡洋舰'
    content = id + "@@@" + "entity" + "@@@" + name + "\n"
    logger.debug(content)
    with open(file, "a+", encoding="utf8")as f:
        data = f.readlines()
    if content not in data:
        data.append(content)
        logger.debug(content)
        f.write(content)


# 更新实体和关系列表用mongo一段时间内的更新添加
def update_entity_relation(COL_NAME, start_time):
    model_col = mydb[COL_NAME]
    # date_time = getdate(7)
    # start_time = convert_time(date_time) * 1000
    end_time = time.time() * 1000
    my_col = model_col.find(
        {"updateTime": {"$gte": start_time, "$lt": end_time}, "checked": "on", "status": "on", "flag": "synced"})
    return my_col


# 更新属性列表用mysql一段时间内的更新添加
def update_property(model_id, start_time):
    db = pymysql.connect(host=gl.MYSQL_CLIENT_NAME, port=3306, user="root", passwd="kgdata", db=gl.MYSQL_DB_NAME,
                         charset='utf8')
    cursor = db.cursor()
    end_time = time.time() * 1000
    # SQL 查询语句
    # sql = "SELECT * FROM ontology_object_property "
    start_time = convert_stamp_time(start_time)
    end_time = time.time()
    end_time = convert_stamp_time(end_time)
    sql = "select * from ontology_property where ontology_id = '{}'and update_time>'{}'and update_time<'{}' ;".format(
        model_id, start_time, end_time)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
    except:
        print("Error: unable to fetch data")

    # 关闭数据库连接
    db.close()
    return results


def update_all_file(model_id, dataPath):
    ENTITY_COL_NAME = model_id + "_entity_instance"
    PROPERTY_COL_NAME = model_id + "_entity_property"
    RELATION_COL_NAME = model_id + "_entity_relation"

    # 待更新的文件
    entity_path = dataPath + "entity_tree.txt"
    user_dict_path = dataPath + "user_dict.txt"
    entity_smi_path = dataPath + "entity.txt"
    entity_el_dict = dataPath + "names.txt"

    property_path = dataPath + "property_tree.txt"
    relation_path = dataPath + "relation_tree.txt"
    relation_sim_file = dataPath + "relation.txt"
    start_time = time.time()

    # 文件最后一次修改的时间
    entity_tree_upate_time = get_file_upate_time(entity_path)
    property_tree_update_time = get_file_upate_time(property_path)
    relation_tree_update_time = get_file_upate_time(relation_path)

    # 更新实体相关文件
    entity_new_col = update_entity_relation(ENTITY_COL_NAME, entity_tree_upate_time)
    for col in entity_new_col:
        entity_name = col["name"]
        entity_id = col["_id"]
        write_file(entity_path, entity_name)
        write_file(user_dict_path, entity_name)
        write_simfile(entity_smi_path, entity_name, "entity")
        write_names(entity_el_dict, entity_id, entity_name)

    # 更新关系相关文件
    relation_new_col = update_entity_relation(RELATION_COL_NAME, relation_tree_update_time)
    for col in relation_new_col:
        relation_name = col["ooName"]
        write_file(relation_path, relation_name)
        write_file(user_dict_path, relation_name)
        write_simfile(relation_sim_file, relation_name, "relation")

    # 更新属性相关文件
    pro_result = update_property(model_id, property_tree_update_time)
    for row in pro_result:
        property_name = row[3]
        write_file(property_path, relation_name)
        write_file(user_dict_path, property_name)
        write_simfile(relation_sim_file, property_name, "property")

    end_time = time.time()
    update_time = end_time - start_time
    logger.debug("文件同步时间:" + str(update_time))


# 用户自定义ruler
def get_custom_ruler(data):
    max_list = ["最快", "最大", "最高", "最强", "最长", "最早"]
    min_list = ["最慢", "最小", "最低", "最矮", "最弱", "最短", "最晚"]
    condition = {"and": "且", "or": "或"}
    comparation = {">": "大于", "<": "小于", ">=": "大于等于", "<=": "小于等于", "!=": "不等于", "=": "等于"}
    patten_dict = {}
    patten_dict_list = []
    # type的顺序列表
    type_list = []
    reg = ""
    # 获取各个类型的正则值
    for component in data:
        componet_type = component["type"]
        content = component["content"]
        if componet_type == "ontology" or componet_type == "entity" or componet_type == "property" or componet_type == "relation":
            if componet_type not in type_list:
                count = 0
            else:
                count = count + 1
            patten_dict[componet_type] = str(count)

        elif componet_type == "minmax":
            if content == "max":
                content = "|".join(max_list)
            else:
                content = "|".join(min_list)
            patten_dict[componet_type] = content
        elif componet_type == "condition":
            content = condition[content]
            patten_dict[componet_type] = content

        elif componet_type == "comparation":
            content = comparation[content]
            patten_dict[componet_type] = content
        elif componet_type == "value":
            content = ".*?"
            patten_dict[componet_type] = content
        else:
            patten_dict[componet_type] = content
        patten_dict_list.append(patten_dict)
        patten_dict = {}
        type_list.append(componet_type)
    print(patten_dict_list)
    # 拼成正则
    for dict_data in patten_dict_list:
        key = list(dict_data.keys())[0]
        value = dict_data[key]
        if key == "ontology":
            value = "<ontology " + value + ">"
        if key == "entity":
            value = "<entity " + value + ">"
        if key == "property":
            value = "<property " + value + ">"
        if key == "relation":
            value = "<relation " + value + ">"

        reg = reg + "(" + value + ")"
        # patten = re.compile(reg)
    return type_list, reg


if __name__ == '__main__':
    update_all_file("af7d3ac99f984442b7960a9e00181b81", "")
    # upate_ruler.update("af7d3ac99f984442b7960a9e00181b81",1626344259783)
    # print(convert_stamp_time(1626344259783/1000))
