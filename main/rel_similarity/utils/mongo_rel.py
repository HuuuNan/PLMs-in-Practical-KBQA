from pymongo import MongoClient

import config.global_var as config
from utils.logger_config import get_logger
import requests
import json

logger = get_logger(config.KBQA_LOG_PATH)
# conn = MongoClient(config.CLIENT_NAME)
# # 连接nlp数据库，没有则自动创建
# db = conn[config.DB_NAME]
# # 模型库
# model_col = db[config.COL_NAME]


# 查看数据库状态
def check_state(model_id):
    result_status = 0
    try:
        model_info = model_col.find_one({'_id': model_id})
        status = str(model_info['trainStatus'])
        if status == '1':
            reuslt = model_info['result']
            result_status = 1
        elif status == '0':
            reuslt = '正在训练中'
            result_status = 0
        else:
            reuslt = '训练失败'
            result_status = -1
    except Exception as e:
        logger.debug(str(e))
        reuslt = '训练失败,请检查'
        result_status = -1
    return reuslt, result_status


# 数据库中录入模型信息
def deal_train_update(model_id, kb_path, parameters):
    model_record = model_col.find({'_id': model_id})
    if model_record.count() == 0:
        model_col.insert({
            '_id': model_id,
            'taskType': 'train',
            'trainStatus': 0,
            'exceptionMessage': None,
            'result': None,
            'kb_id': kb_path,
            'kb_path': kb_path,
            'parameters': parameters,
        })
        status = 1
        return model_id, status
    else:
        model_col.update({"_id": model_id},
                         {'$set': {
                             '_id': model_id,
                             'taskType': 'train',
                             'trainStatus': 0,
                             'exceptionMessage': None,
                             'result': None,
                             'kb_id': kb_path,
                             'kb_path': kb_path,
                             'parameters': parameters,
                         }})
        status = 1
        return model_id, status


# 更新数据库中模型得状态
def model_col_update(model_id, train_status, message, result):
    model_col.update({"_id": model_id},
                     {'$set': {
                         '_id': model_id,
                         'trainStatus': train_status,
                         'exceptionMessage': message,
                         'result': result

                     }})
    status = 1
    return model_id, status


# 查询数据库中所有模型得状态，强制停止得改成训练失败
def find_all_model_status():
    for x in model_col.find():
        status = str(x['trainStatus'])
        model_id = str(x["_id"])
        if status == "0":
            status = -1
            model_col.update({"_id": model_id},
                             {'$set': {
                                 'trainStatus': status,
                                 'exceptionMessage': "被强制停止",
                             }})


# 数据库中插入模板
def insert_custom_ruler(model_id, action, templateId, templateType, templateStatus, typeOrder, patten):
    ruler_col = db[config.RULER_COL_NAME + "_" + model_id]
    if action == "insert":
        model_record = ruler_col.find({'_id': templateId})
        if model_record.count() == 0:

            ruler_col.insert({
                '_id': templateId,
                'templateType': templateType,
                'templateStatus': templateStatus,
                'typeOrder': typeOrder,
                'patten': patten
            })
            status = 0
            return model_id, status
        else:
            ruler_col.update({'_id': templateId},
                             {'$set': {
                                 '_id': templateId,
                                 'templateType': templateType,
                                 'templateStatus': templateStatus,
                                 'typeOrder': typeOrder,
                                 'patten': patten
                             }})
            status = 0
        return model_id, status
    elif action == "update":
        ruler_col.update({'_id': templateId},
                         {'$set': {
                             '_id': templateId,
                             'templateType': templateType,
                             'templateStatus': templateStatus,
                             'typeOrder': typeOrder,
                             'patten': patten
                         }})
        status = 0
    else:
        status = 2
    return model_id, status


# 数据库中更新模板状态
def upate_ruler_status(model_id, templateId, templateStatus):
    ruler_col = db[config.RULER_COL_NAME + "_" + model_id]
    ruler_update = ruler_col.find_one({'_id': templateId})["_id"]
    if ruler_update:
        ruler_col.update({'_id': templateId},
                         {'$set': {
                             '_id': templateId,
                             'templateStatus': templateStatus,
                         }})
        status = 0
    else:
        status = 1

    return status


# 数据库中删除模板
def delete_custom_ruler(model_id, templateId):
    ruler_col = db[config.RULER_COL_NAME + "_" + model_id]
    ruler_delete = ruler_col.find_one({'_id': templateId})["_id"]
    if ruler_delete:
        ruler_col.delete_one({'_id': templateId})
        status = 0
    else:
        status = 1
    return status


# 写入同义词词表
def synonyms_form(model_id, name, type, synonyms):
    # 同义词库
    synonyms_col = db[config.SYNONYMS_COL_NAME + "_" + model_id]
    if synonyms_col.count_documents({'name': name}) == 0:
        synonyms_col.insert_one({
            'name': name,
            'type': type,
            'synonyms': synonyms,

        })
        status = 1
        return model_id, status
    else:
        # print(name,type,synonyms)
        synonyms_col.update_one({"name": name},
                                {'$set': {
                                    'name': name,
                                    'type': type,
                                    'synonyms': synonyms,

                                }})
        status = 1
        return model_id, status


# 查同义词表
def find_synonyms(model_id, alias):
    synonyms_col = db[config.SYNONYMS_COL_NAME + "_" + model_id]
    i = synonyms_col.find({"synonyms": alias})
    name = i[0]["name"]
    sim_type = i[0]["type"]

    return name, sim_type


if __name__ == '__main__':
    synonyms_form("1", "zhou2", "entty", ["zhou2", "zhouxiaoning"])
    find_synonyms("1", "zhou2")
