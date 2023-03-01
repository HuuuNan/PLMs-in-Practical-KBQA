# 部署 侦察 返航 加油 打击行动 军事行动 军事训练 特种作战行动 空中预警 情报支援 验证测试 运输

# 输出事件、机型、航班、任务、军种、编号、起始地、目的地。

import json
import re
from models.el.el_predict import entity_linking

types = ["部署", "侦察", "返航", "加油", "打击行动", "军事行动", "军事训练", "特种作战行动", "空中预警", "情报支援", "验证测试", "运输"]
types_dict={
    "部署":["部署"],
    "侦察":["侦察"],
    "返航":["返航","返回"],
    "加油":["加油"],
    "打击行动":["打击行动"],
    "军事行动":["军事行动"],
    "军事训练":["军事训练"],
    "特种作战行动":["特种作战行动","特种作战"],
    "空中预警":["空中预警"],
    "情报支援":["情报支援"],
    "验证测试":["验证测试"],
    "运输":["运输"]
}

# 提取军种
def army_services(text_length, con):
    services = ["海军", "陆军", "空军", "海岸警卫队", "自卫队"]
    se_dict = {}
    for s in services:
        if s in con:
            se_dict = {'argument_name': s, "argument_type": "军种", "argument_id": "12314",
                       "argument_start": text_length + con.index(s),
                       "argument_end": text_length + con.index(s) + len(s)}
            break
    if se_dict == {}:
        se_dict = {'argument_name': None, "argument_type": "军种", "argument_id": "12314",
                   "argument_start": None, "argument_end": None}
    return se_dict


# 正则提取起始地
def starting_place(text_length, con):
    op_pattern = re.compile('[从|由](.*?)[起飞|出港|进入]')
    op = re.findall(op_pattern, con)
    if op != []:
        op_dict = {'argument_name': op[0], "argument_type": "起始地", "argument_id": "12314",
                   "argument_start": text_length + con.index(op[0]),
                   "argument_end": text_length + con.index(op[0]) + len(op[0])}
    else:
        op_dict = {'argument_name': None, "argument_type": "起始地", "argument_id": "12314",
                   "argument_start": None, "argument_end": None}
    return op_dict


# 正则提取目的地
def destination(text_length, con):
    de_pattern = re.compile('[降落在|往|进入|在|向]+([\u4e00-\u9fa5]*?)[，|上空|开展|部署|对]+')
    de = re.findall(de_pattern, con)
    if de != []:
        de_dict = {'argument_name': de[0], "argument_type": "目的地", "argument_id": "12314",
                   "argument_start": text_length + con.index(de[0]),
                   "argument_end": text_length + con.index(de[0]) + len(de[0])}
    else:
        de_dict = {'argument_name': None, "argument_type": "目的地", "argument_id": "12314",
                   "argument_start": None, "argument_end": None}
    return de_dict


# 正则提取编号/机型/航班  (编号为15-5822的C-130J运输机（HKY799）)
def serial_aircraft_flight(ee_dict, text_length, con, se_dict, op_dict, de_dict, type):
    at_pattern = re.compile('编号为([0-9,-]*)的([A-Z]+[/,0-9,A-Z,-]+[\u4e00-\u9fa5]+机)[\(|（]([A-Z,0-9,\,,、]+)[(\)|）]')

    at_pattern2 = re.compile('海军([\u4e00-\u9fa5]*舰)([A-Z,a-z,\s,\。]+)[\(|（]([A-Z,0-9,\,,、,\s]+)[(\)|）]')
    at = re.findall(at_pattern, con)
    at2 = re.findall(at_pattern2, con)
    get = False
    for a in at:
        event_dict = {}
        event_dict['event_name'] = type
        event_dict['source'] = con
        event_dict['arguments'] = []
        argument_dict = {'argument_name': a[1], "argument_type": "机型（船型）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[1]),
                         "argument_end": text_length + con.index(a[1]) + len(a[1])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': a[0], "argument_type": "编号（船名称）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[0]),
                         "argument_end": text_length + con.index(a[0]) + len(a[0])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': a[2], "argument_type": "航班（弦号）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[2]),
                         "argument_end": text_length + con.index(a[2]) + len(a[2])}
        event_dict['arguments'].append(argument_dict)
        event_dict['arguments'].append(se_dict)
        event_dict['arguments'].append(op_dict)
        event_dict['arguments'].append(de_dict)


        ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True
    for a in at2:
        event_dict = {}
        event_dict['event_name'] = type
        event_dict['source'] = con
        event_dict['arguments'] = []
        argument_dict = {'argument_name': a[0], "argument_type": "机型（船型）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[0]),
                         "argument_end": text_length + con.index(a[0]) + len(a[0])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': a[1], "argument_type": "编号（船名称）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[1]),
                         "argument_end": text_length + con.index(a[1]) + len(a[1])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': a[2], "argument_type": "航班（弦号））", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[2]),
                         "argument_end": text_length + con.index(a[2]) + len(a[2])}
        event_dict['arguments'].append(argument_dict)
        event_dict['arguments'].append(se_dict)
        event_dict['arguments'].append(op_dict)
        event_dict['arguments'].append(de_dict)


        ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True
    return ee_dict, get


# 正则提取机型/航班     (RQ-4B全球鹰无人机（FORTE10）)
def aircraft_flight(ee_dict, text_length, con, se_dict, op_dict, de_dict, type):
    at_pattern = re.compile('([A-Z]+[/,0-9,A-Z,-]+[\u4e00-\u9fa5]+机)[\(|（]([A-Z,a-z,0-9,\,,、]+)[(\)|）]')
    at = re.findall(at_pattern, con)
    get = False
    for a in at:
        for hb in a[1].split('、'):
            event_dict = {}
            event_dict['event_name'] = type
            event_dict['source'] = con
            event_dict['arguments'] = []
            argument_dict = {'argument_name': a[0], "argument_type": "机型（船型）", "argument_id": "12314",
                             "argument_start": text_length + con.index(a[0]),
                             "argument_end": text_length + con.index(a[0]) + len(a[0])}
            event_dict['arguments'].append(argument_dict)
            argument_dict = {'argument_name': None, "argument_type": "编号（船名称）", "argument_id": "12314",
                             "argument_start": None,
                             "argument_end": None}
            event_dict['arguments'].append(argument_dict)
            argument_dict = {'argument_name': hb, "argument_type": "航班（弦号）", "argument_id": "12314",
                             "argument_start": text_length + con.index(hb),
                             "argument_end": text_length + con.index(hb) + len(hb)}
            event_dict['arguments'].append(argument_dict)
            event_dict['arguments'].append(se_dict)
            event_dict['arguments'].append(op_dict)
            event_dict['arguments'].append(de_dict)


            ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True
    return ee_dict, get


# 正则提取机型/航班/编号  (KC-135R加油机（GOLD41、42，编号62-3526、61-0292）)
def aircraft_flight_serial(ee_dict, text_length, con, se_dict, op_dict, de_dict, type):
    at_pattern = re.compile('([A-Z]+[/,0-9,A-Z,-]+[\u4e00-\u9fa5]+机)[\(|（](.+)，编号(.+)[(\)|）]')
    at = re.findall(at_pattern, con)
    get = False
    for a in at:
        for hb, bh in zip(a[1].split('、'), a[2].split('、')):
            event_dict = {}
            event_dict['event_name'] = type
            event_dict['source'] = con
            event_dict['arguments'] = []
            argument_dict = {'argument_name': a[0], "argument_type": "机型（船型）", "argument_id": "12314",
                             "argument_start": text_length + con.index(a[0]),
                             "argument_end": text_length + con.index(a[0]) + len(a[0])}
            event_dict['arguments'].append(argument_dict)
            argument_dict = {'argument_name': bh, "argument_type": "编号（船名称）", "argument_id": "12314",
                             "argument_start": text_length + con.index(bh),
                             "argument_end": text_length + con.index(bh) + len(bh)}
            event_dict['arguments'].append(argument_dict)
            argument_dict = {'argument_name': hb, "argument_type": "航班（弦号）", "argument_id": "12314",
                             "argument_start": text_length + con.index(hb),
                             "argument_end": text_length + con.index(hb) + len(hb)}
            event_dict['arguments'].append(argument_dict)
            event_dict['arguments'].append(se_dict)
            event_dict['arguments'].append(op_dict)
            event_dict['arguments'].append(de_dict)


            ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True
    return ee_dict, get


# 正则提取编号/机型     (编号为58-0086的KC-135R加油机)
def serial_aircraft(ee_dict, text_length, con, se_dict, op_dict, de_dict, type):
    at_pattern = re.compile('编号为([0-9,-]*)的([A-Z]+[/,0-9,A-Z,-]+[\u4e00-\u9fa5]+机)')
    at = re.findall(at_pattern, con)
    get = False
    for a in at:
        event_dict = {}
        event_dict['event_name'] = type
        event_dict['source'] = con
        event_dict['arguments'] = []
        argument_dict = {'argument_name': a[1], "argument_type": "机型（船型）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[1]),
                         "argument_end": text_length + con.index(a[1]) + len(a[1])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': a[0], "argument_type": "编号（船名称）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a[0]),
                         "argument_end": text_length + con.index(a[0]) + len(a[0])}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': None, "argument_type": "航班（弦号）", "argument_id": "12314",
                         "argument_start": None,
                         "argument_end": None}
        event_dict['arguments'].append(argument_dict)
        event_dict['arguments'].append(se_dict)
        event_dict['arguments'].append(op_dict)
        event_dict['arguments'].append(de_dict)

        ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True

    return ee_dict, get


# 正则提取机型    (P-8A海王海上反潜巡逻机)
def aircraft(ee_dict, text_length, con, se_dict, op_dict, de_dict, type):
    at_pattern = re.compile('([A-Z]+[/,0-9,A-Z,-]+[“,”,\u4e00-\u9fa5]+机)')
    at = re.findall(at_pattern, con)
    get = False
    for a in at:
        event_dict = {}
        event_dict['event_name'] = type
        event_dict['source'] = con
        event_dict['arguments'] = []
        argument_dict = {'argument_name': a, "argument_type": "机型（船型）", "argument_id": "12314",
                         "argument_start": text_length + con.index(a),
                         "argument_end": text_length + con.index(a) + len(a)}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': None, "argument_type": "编号（船名称）", "argument_id": "12314",
                         "argument_start": None,
                         "argument_end": None}
        event_dict['arguments'].append(argument_dict)
        argument_dict = {'argument_name': None, "argument_type": "航班（弦号）", "argument_id": "12314",
                         "argument_start": None,
                         "argument_end": None}
        event_dict['arguments'].append(argument_dict)
        event_dict['arguments'].append(se_dict)
        event_dict['arguments'].append(op_dict)
        event_dict['arguments'].append(de_dict)

        ee_dict['data']['event'].append(event_dict)
        ee_dict['retCode'] = 0
        get = True

    return ee_dict, get


def start_match(line):
    ee_dict = {}
    ee_dict['retCode'] = 1
    ee_dict['data'] = {}
    ee_dict['data']['text'] = line.strip()
    ee_dict['data']['event'] = []
    ee_dict['message'] = None
    ee_dict['exceptionId'] = None

    for j, con in enumerate(line.split('。')):
        for type in types:
            type_true = False
            for t in types_dict[type]:
                if t in con:
                    type_true = True
                    # print(type)
                    text_length = len('。'.join(line.split('。')[:j]))
                    # 军种
                    se_dict = army_services(text_length, con)

                    # 起始地
                    op_dict = starting_place(text_length, con)

                    # 目的地
                    de_dict = destination(text_length, con)

                    # 编号/机型/航班
                    ee_dict, get = serial_aircraft_flight(ee_dict, text_length, con, se_dict, op_dict, de_dict, type)
                    if not get:
                        ee_dict, get = aircraft_flight(ee_dict, text_length, con, se_dict, op_dict, de_dict, type)
                    if not get:
                        ee_dict, get = aircraft_flight_serial(ee_dict, text_length, con, se_dict, op_dict, de_dict, type)
                    if not get:
                        ee_dict, get = serial_aircraft(ee_dict, text_length, con, se_dict, op_dict, de_dict, type)
                    if not get:
                        ee_dict, get = aircraft(ee_dict, text_length, con, se_dict, op_dict, de_dict, type)
                    break
            if type_true:
                break

    event = ee_dict["data"]["event"]
    text = ee_dict["data"]["text"]
    for e in event:
        arguments = e["arguments"]
        for a in arguments:
            argument_name = a["argument_name"]
            argument_type = a["argument_type"]
            print(argument_name)
            if argument_name != None:
                mids = entity_linking([argument_name], 5)
                if mids:
                    for mid in mids[0]:
                        mid_id = mid[0][0]
                        mid_name = mid[0][1]
                        mid_type = mid[0][2]
                        # if mid_type == argument_type:
                        # a["argument_id"] = mid_id
                        # if argument_name != mid_name and mid_name in text:
                        #     a["argument_name"] = mid_name
                        # break
                        if argument_type == "起始地" or argument_type == "目的地":
                            if mid_type == "基地" or mid_type == "null":
                                print(mid_name,mid_id)
                                a["argument_id"] = mid_id
                                break
                        if argument_type == "机型（船型）":
                            if mid_type == "飞机" or mid_type == "null" or mid_type == "舰船":
                                a["argument_id"] = mid_id
                                print(mid_name, mid_id)
                                break
                        if argument_type == "军种":
                            if mid_type == "军队" or mid_type == "null":
                                a["argument_id"] = mid_id
                                print(mid_name, mid_id)
                                break

                else:
                    a["argument_id"] = None
            else:
                a["argument_id"] = None

            # print(mids[0][0][])

    return ee_dict


if __name__ == '__main__':
    example_data = '3月23日澳大利亚1架编号为A41-207的C-17运输机（ASY460）从布里斯班起飞，降落在巴布亚新几内亚莫瑞斯比港，给该国运送了3箱疫苗和一些食品，跟打发叫花子一样。​​​​​​​​'

    ee_dict = start_match(example_data)
    # mids = entity_linking("导弹驱逐舰", 3)
    # print(mids)
    print(ee_dict)

