import json

import random
import jieba
import re


def cleandata(text):
    text = re.sub('"', '', text)
    return text


def create_traindata(datapath, ontology_dict):
    f_out = open(datapath + "data.txt", "w", encoding="utf8")
    f_Rel = open(datapath + "relation.txt", "r", encoding="utf8")

    rel_list = []
    rels = f_Rel.readlines()
    for rel in rels:
        rel = rel.strip().split("\t")[0]
        rel_list.append(rel)

    f_real = open(datapath + "real_data.txt", "r", encoding="utf8")
    real_data = f_real.readlines()
    for real in real_data:
        real_index = real.strip().split("\t")
        real_type = real_index[3]
        if real_type in rel_list:
            f_out.write(real)

    f = open(datapath + "triples.txt", "r", encoding="utf8")
    lines = f.readlines()
    n = 1
    # place = ["科研机构", "科研院所", "神州云台", "机场", "目标", "地理", "国家", "行政区划"]
    # person = ["政界", "文化艺术界"]
    place = ontology_dict["地理"] + ontology_dict["军事对象"]
    person = ["人"]
    danwei = ['海里',
              '英里',
              '/节',
              'km/节',
              '吨',
              '-吨',
              '公里',
              '公里/节',
              '公里/小时'
              '海里节',
              '海里，节',
              '海里/节',
              '海哩/节',
              '海浬/节',
              '毫米',
              '节',
              '节/海里',
              '节海里',
              '节行驶英里',
              '节下海里',
              '克',
              '里',
              '里/节',
              '米',
              '千克',
              '千米',
              '千米/节',
              '千米/时',
              '千米/小时',
              '千米每小时',
              '万海里/节',
              '英里，节',
              '英里/节',
              '余英里',
              '约海里',
              '最大海里',
              '厘米',
              '分米',
              '人',
              '位',
              "mm",
              "cm",
              "个"]
    # if len(lines) > 100000:
    #
    #     data_num = len(lines)
    # else:
    data_num = 100000
    n=0
    for type in ontology_dict.keys():

        if len(ontology_dict[type])>0:
            print(type,ontology_dict[type])
            print(n)
            n = n + 1
            for type_num in range(8000):

                '''总共4个类别
                类别0是2个实体找关系
                类别1是头实体关系找尾实体
                类别2是单个实体
                类别3是关系尾实体找头实体'''

                quetion_type = [0, 1, 2, 3]
                data = ""
            # for i in range(data_num):
            #     if data_num == len(lines):
            #         chosen_one = lines[i].strip().replace("\t", "").replace("\n", "")
            #     else:
                metion_type=""
                while(metion_type not in ontology_dict[type]):
                    chosen_one = random.choice(lines).strip().replace("\t", "").replace("\n", "")

                    chosen_one = cleandata(chosen_one)
                    index = chosen_one.split("@@@")

                    id = index[0]

                    metion = index[1]
                    metion = ''.join(metion.split())
                    metion_type = index[4]

                    rel = index[2]
                    rel = ''.join(rel.split())
                    rel_type = index[5]

                    answer = index[3]
                    answer = ''.join(answer.split())
                    answer_list = jieba.lcut(answer)
                    answer_type = index[6]
                print(metion_type)
                type_chose = random.choice(quetion_type)
                question = ""
                if rel == "重量":
                    question = metion + "有多重"
                elif "大小" in rel or "体积" == "rel" or "速度" in rel:
                    question = metion + "有多大"
                elif rel == "高度":
                    question = metion + "有多高"
                elif rel == "部署":
                    question = metion + "部署了什么"
                elif rel == "包含":
                    question = metion + "有什么"
                elif rel == "研制单位":
                    question = metion + "是在哪里研制的"
                elif rel == "研制时间":
                    question = metion + "什么时间研制"
                if rel_type == "relation":
                    question = metion + rel + "的" + answer_type
                else:

                    question = metion + "的" + rel
                if question:

                    label_B = question.index(metion)
                    label = ["O" for i in range(len(question))]
                    label[label_B] = "B"
                    label[label_B + 1:len(metion)] = ["I" for i in range(len(metion) - 1)]

                    if metion and rel and answer and question:
                        data = str(
                            n) + "\t" + id + "\t" + metion + "\t" + rel + "\t" + answer + "\t" + question + "\t" + " ".join(
                            label)
                if type_chose == 0:
                    chosen_two = random.choice(lines).strip().replace("\t", "").replace("\n", "")
                    chosen_two = cleandata(chosen_two)
                    index = chosen_two.split("@@@")

                    id2 = index[0]
                    # print(index)
                    metion2 = index[1]
                    metion2 = ''.join(metion2.split())
                    rela_predix = random.choice(["的关系是什么", "有什么关系", "的关系"])

                    question = metion + "和" + metion2 + rela_predix

                    label_B = question.index(metion)
                    label = ["O" for i in range(len(question))]
                    label[label_B] = "B"
                    label[label_B + 1:len(metion)] = ["I" for i in range(len(metion) - 1)]

                    label_B2 = question.index(metion2)
                    label[label_B2] = "B"
                    label[label_B2 + 1:len(metion)] = ["I" for i in range(len(metion2) - 1)]

                    if metion and metion2 and question:
                        data = str(
                            n) + "\t" + id + " " + id2 + "\t" + metion + " " + metion2 + "\t" + "关系" + "\t" + "null" + "\t" + question + "\t" + " ".join(
                            label)
                elif type_chose == 1:
                    flag = 0
                    for d in answer_list:
                        if d in danwei:
                            flag = 1
                        elif "年" in d or "月" in d or "日" in d:
                            flag = 2

                        else:
                            flag = 0
                    prefix = random.choice([0, 1, 2])
                    mention_place_prefix = random.choice(["在哪里", "在什么地方", "在什么单位"])
                    mention_person_prefix = random.choice(["是谁的", "是哪位的"])
                    middle_prefix = random.choice(["的", ""])
                    if prefix == 0:
                        if flag == 1:
                            question = metion + "的" + rel + "是多少"
                        elif flag == 2:
                            question = metion + "的" + rel + "是什么时候"
                        elif flag == 0:
                            question = metion + "的" + rel + "是什么"
                        # elif answer_type in place:
                        #     question = "哪里是"+metion+"的"+rel
                        # elif answer_type in person:
                        #     question = metion + "的" + rel + "是谁"

                    elif prefix == 1:
                        question = metion + "的" + rel
                    else:
                        if flag == 2:
                            question = metion + "是什么时候" + rel.replace("时间", "").replace("年代", "").replace("日期", "")

                        elif answer_type in place and rel_type == "relation":
                            question = metion + mention_place_prefix + rel.replace("单位", "") + "的"
                        elif answer_type in person:
                            question = metion + mention_person_prefix + rel
                        else:
                            question = metion + middle_prefix + rel

                    # if question:
                    # print("++++++++++", question, metion)
                    label_B = question.index(metion)
                    # print(label_B)
                    label = ["O" for i in range(len(question))]
                    # print(label)
                    if label and len(label) >= label_B:
                        label[label_B] = "B"
                        label[label_B + 1:len(metion)] = ["I" for i in range(len(metion) - 1)]

                    if metion and rel and answer and question:
                        data = str(
                            n) + "\t" + id + "\t" + metion + "\t" + rel + "\t" + answer + "\t" + question + "\t" + " ".join(
                            label)
                elif type_chose == 2:
                    question = metion
                    label_B = question.index(metion)
                    label = ["O" for i in range(len(question))]
                    if label and len(label) >= label_B:
                        label[label_B] = "B"
                        label[label_B + 1:len(metion)] = ["I" for i in range(len(metion) - 1)]
                        data = str(
                            n) + "\t" + id + "\t" + metion + "\t" + "null" + "\t" + "null" + "\t" + metion + "\t" + " ".join(
                            label)
                elif type_chose == 3:
                    flag = 0
                    for d in answer_list:
                        if d in danwei:
                            flag = 1
                        elif "年" in d or "月" in d or "日" in d:
                            flag = 2
                        else:
                            flag = 0
                    prefix = random.choice([0, 1, 2])
                    mention_place_prefix = random.choice(["哪里", "什么地方", "什么单位"])
                    mention_person_prefix = random.choice(["哪些人是", "谁是"])
                    # middle_prefix = random.choice(["哪些被", "什么被"])
                    if prefix == 0:
                        if flag == 1:
                            question = metion + "的" + rel + "是多少"
                        elif flag == 2:
                            question = metion + "什么时间" + rel
                        # elif flag == 0:
                        #     question = "哪些" + metion + "被" + rel

                    elif prefix == 1:
                        question = "什么是" + metion + "的" + rel
                    else:
                        if flag == 2:
                            question = metion + "什么时候" + rel

                        elif answer_type in place:
                            question = mention_place_prefix + "是" + metion + rel
                        elif answer_type in person:
                            question = mention_person_prefix + metion + "的" + rel
                        elif answer_type in place:
                            question = middle_prefix + rel + "在" + metion

                    if question:
                        label_B = question.index(metion)
                        label = ["O" for i in range(len(question))]
                        label[label_B] = "B"
                        label[label_B + 1:len(metion)] = ["I" for i in range(len(metion) - 1)]

                        if metion and rel and answer and question:
                            data = str(
                                n) + "\t" + id + "\t" + metion + "\t" + rel + "\t" + answer + "\t" + question + "\t" + " ".join(
                                label)
                # print(data)
                if data:
                    f_out.write(data + "\n")
                    n += 1


if __name__ == '__main__':
    create_traindata("")
