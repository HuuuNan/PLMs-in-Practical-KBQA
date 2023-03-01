import csv
import json
import os
import pickle
import random
import re

import jieba
from tqdm import tqdm
from utils import mongo_rel

def clean_data(mystr):
    mystr = mystr.replace("\r", "", 1000).replace("\r\r", "", 1000).replace("\n", "").replace(".", "-").strip()
    return mystr


def read_kb_file(kb_file, datapth):
    f_entity = open(datapth + "entity_tree.txt", "w", encoding="utf8")
    f_ontology = open(datapth + "ontology_tree.txt", "w", encoding="utf8")
    f_property = open(datapth + "property_tree.txt", "w", encoding="utf8")
    f_relation = open(datapth + "relation_tree.txt", "w", encoding="utf8")
    entity_tree = []
    ontology_tree = []
    property_tree = []
    relation_tree = []
    kb_data = []
    with open(kb_file, "r", encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            kb_data = kb_data + data
        # 读取三元组文件
        # while True:
        #     line = f.readline()
        #     if line:
        #         data = json.loads(line)
        #         kb_data = kb_data + data
        #     else:
        #         break

    ontology_dict = {}
    instance_dict = {}
    print("数据处理中。。。。。")
    for kb_json in tqdm(kb_data):
        id = kb_json["sourceId"]
        entity = clean_data(kb_json["sourceName"])
        entity_type = kb_json["sourceType"]
        rel = clean_data(kb_json["relationName"])
        rel_type = kb_json["relationType"]
        target = clean_data(kb_json["targetName"])
        target_type = kb_json["targetType"]
        if rel_type == "property":
            answer_type = "property"
            if rel not in property_tree:
                # mongo_rel.synonyms_form(model_id,rel,rel_type,)
                property_tree.append(rel)
        else:
            answer_type = target_type
            if rel not in relation_tree:
                relation_tree.append(rel)
        if target_type == "ontology":
            if target not in ontology_dict.keys():
                ontology_dict[target] = []
                if target not in ontology_tree:
                    ontology_tree.append(target)

        if entity_type == "ontology":
            if entity not in ontology_tree:
                ontology_tree.append(entity)

            if entity not in ontology_dict.keys():
                ontology_dict[entity] = [(rel, rel_type, answer_type)]

            else:
                if (rel, rel_type, answer_type) not in ontology_dict[entity]:
                    ontology_dict[entity].append((rel, rel_type, answer_type))
        else:
            if entity not in ontology_tree:
                entity_tree.append(entity)

        if rel == "别名":
            alias = target.rstrip(";").split(";")
        else:
            alias = None
        if rel == "摘要":
            desc = target
        else:
            desc = None

        instance_dict[id] = (entity, entity_type, alias, desc)
    # print(instance_dict)
    f_entity.write("\n".join(entity_tree))
    f_ontology.write("\n".join(ontology_tree))
    f_relation.write("\n".join(relation_tree))
    f_property.write("\n".join(property_tree))
    return ontology_dict, instance_dict


# spacy的kb数据构建
def make_entities(instance_dict, datapath):
    def get_name(subject):
        if "(" in subject and ")" in subject:
            index_s = subject.index("(")
            index_e = subject.index(")")
            if "(" != subject[0]:
                name = subject[0:index_s]
            else:
                name = subject[index_s + 1:index_e]
        elif "(" in subject or ")" in subject:
            name = subject.replace(")", "").replace("(", "")
        else:
            name = subject
        return name

    kb_csv = datapath + "entities.csv"
    fout = open(kb_csv, "w", encoding="utf8", newline="")
    csv_writer = csv.writer(fout)
    print("==================生成spacy kb=======================")
    for id in tqdm(instance_dict.keys()):
        entity, entity_type, alias, desc = instance_dict[id]
        # print(id, entity, alias, desc)
        if alias == None:
            alias = [entity]
        if desc == None:
            desc = entity
        for a in alias:
            # 4. 写入csv文件内容
            csv_writer.writerow([id, entity, a, desc, entity_type])


def reach_index(ontology_dict, instance_dict, outpath):
    print("creating the reach.pkl index map...")
    index = {}
    size = 0
    for instance_id in tqdm(instance_dict.keys()):
        subject = instance_id

        ontology_type = instance_dict[instance_id][1]
        if ontology_type != "ontology":
            predicate_list = ontology_dict[ontology_type]
        else:
            ontology_type = instance_dict[instance_id][0]
            predicate_list = ontology_dict[ontology_type]
        for predicate_info in predicate_list:
            predicate, type, answer_type = predicate_info
            if index.get(subject) is not None:
                index[subject].add(predicate)
            else:
                index[subject] = set([predicate])

        size += 1
    # print(index)

    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")


def create_inverted_index_entity(instance_dict, outpath):
    def get_all_ngrams(tokens):
        all_ngrams = set()
        # max_n = min(len(tokens), 3)
        max_n = len(tokens)
        for n in range(1, max_n + 1):
            ngrams = find_ngrams(tokens, n)
            all_ngrams = all_ngrams | ngrams
        return all_ngrams

    def find_ngrams(input_list, n):
        ngrams = zip(*[input_list[i:] for i in range(n)])
        return set(ngrams)

    def get_name_ngrams(entity_name):
        processed_name = entity_name  # lowercase the name
        # name_tokens = processed_name.split()
        name_tokens = list(jieba.cut(processed_name))
        # print(name_tokens)
        name_ngrams = get_all_ngrams(name_tokens)

        return name_ngrams

    print("creating the entity.pkl index map...")
    index = {}
    size = 0
    # with open(namespath, 'r', encoding="utf8") as f:
    for entity_mid in tqdm(instance_dict.keys()):
        entity_name, entity_type, alias, desc = instance_dict[entity_mid]

        r1 = "[^\u4e00-\u9fa5^a-z^A-Z^0-9]"

        name_ngrams = get_name_ngrams(re.sub(r1, "", entity_name).lower())
        # print(name_ngrams)
        for ngram_tuple in name_ngrams:
            size += 1

            ngram = "".join(ngram_tuple)
            # print(ngram)
            if index.get(ngram) is not None:
                index[ngram].add((entity_mid, entity_name, entity_type))
            else:
                index[ngram] = set([(entity_mid, entity_name, entity_type)])

    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    print("dumping to pickle...")
    with open(outpath, 'wb') as f:
        # print(index)
        pickle.dump(index, f)
    print("DONE")


class train_data_generator():
    def __init__(self, datapath, ontology_dict, ontology_info, instance_dict, data_num):

        self.ontology_info = ontology_info
        self.ontology_list = list(ontology_info.keys())
        self.instance_dict = instance_dict
        self.entity_list = list(instance_dict.keys())

        '''总共4个类别
        类别0是2个实体找关系
        类别1是头-尾-疑问词
        类别2是单个实体
        类别3是疑问词-头-尾
        类别4是头-疑问词-尾'''
        self.quetion_type = [0, 1, 2, 3, 4]

        # 疑问词 when where what who
        self.common_question_word = ["什么", "啥"]
        self.people_question_word = ["什么名字", "谁", "哪位", "什么"]
        self.place_question_word = ["哪里", "什么单位", "哪", "什么地方"]
        self.time_question_word = ["什么时候", "多会儿", "何时"]
        self.guanxi_question_word = ["的关系是什么", "有什么关系", "的关系"]

        # 单位
        self.sudu_danwei = ['公里', '公里/节', '公里/小时', '海里', '英里', 'km/h', "m/h", '千米/时', '千米/小时'
            , '千米每小时', '余英里', '约海里', '最大海里', "节"]
        self.zhongliang_danwei = ['吨', '克', '千克', "g", "kg", "t"]
        self.tiji_danwei = ["平方厘米", "平方分米", "平方米", "公顷", "平方千米", "立方厘米", "立方分米", "立方米"]
        self.changdu_danwei = ["mm", "cm", "km", "微米", "毫米", "厘米", "分米", "米", "十米", "百米", "千米"]
        self.time_danwei = ["年", "月", "日", "世纪", "年代"]
        self.liangci_danwei = ["人", "位"]
        self.duliang_danwei = self.sudu_danwei + self.zhongliang_danwei + self.tiji_danwei + self.changdu_danwei

        self.f_out = open(datapath + "data.txt", "w", encoding="utf8")
        f_Rel = open(datapath + "relation_tree.txt", "r", encoding="utf8")
        f_Pro = open(datapath + "property_tree.txt", "r", encoding="utf8")
        # 加载属性关系列表
        self.rel_list = []
        rels = f_Rel.readlines()
        pros = f_Pro.readlines()
        for rel in rels:
            rel = rel.strip()
            self.rel_list.append(rel)
        for pro in pros:
            pro = pro.strip()
            self.rel_list.append(pro)

        # 添加一些真实的百科简单问句
        f_real = open(datapath + "real_data.txt", "r", encoding="utf8")
        real_data = f_real.readlines()
        for real in real_data:
            real_index = real.strip().split("\t")
            real_type = real_index[3]
            if real_type in self.rel_list:
                self.f_out.write(real)

        # 根据三元组文件来生成训练数据
        # f = open(datapath + "triples.txt", "r", encoding="utf8")
        # self.lines = f.readlines()

        # 根据类型加疑问词
        self.ontology_dict = ontology_dict
        self.types = ontology_dict.keys()
        if "地理" in ontology_dict.keys() and "军事对象" in ontology_dict.keys():
            self.place = ontology_dict["地理"] + ontology_dict["军事对象"]
        else:
            self.place = []
        if "人" in ontology_dict.keys():
            self.person = ontology_dict["人"]
        else:
            self.person = []

        self.metion = ""
        self.id = ""
        self.metion_type = ""

        self.rel = ""
        self.rel_type = ""

        self.answer = ""
        self.answer_type = ""

        self.metion2 = ""
        self.id2 = ""
        self.datanum = data_num

    # 数据清洗
    def cleandata(self, text):
        text = re.sub('"', '', text)
        return text

    # 类别0是2个实体找关系
    def type1(self, n):
        data = ""
        question = self.metion + "和" + self.metion2 + random.choice(self.guanxi_question_word)

        if self.metion and self.metion2:
            data = str(
                n) + "\t" + self.id + " " + self.id2 + "\t" + self.metion + " " + self.metion2 + "\t" + "关系" + "\t" + "null" + "\t" + question + "\t" + self.metion_type
        return data

    # 是头-尾-疑问词
    def type2(self, target_type, n):
        data = ""
        if target_type == "数字":
            question = self.metion + "的" + self.rel + "是多少"
        elif target_type == "时间":
            question = self.metion + "的" + self.rel + "是" + random.choice(self.time_question_word)
        elif self.answer_type in self.person:
            question = self.metion + "的" + self.rel + "是" + random.choice(self.people_question_word)
        elif self.answer_type in self.place:
            question = self.metion + "的" + self.rel + "是" + random.choice(
                self.place_question_word)
        else:
            question = self.metion + "的" + self.rel + "是什么"
        if self.metion and self.rel:
            data = str(
                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question + "\t" + self.metion_type
        return data

    # 疑问词-头-尾
    def type4(self, target_type, n):
        data = ""
        if self.answer_type in self.place:
            question = random.choice(self.place_question_word) + "是" + self.metion + "的" + self.rel

        elif self.answer_type in self.person:
            question = random.choice(self.people_question_word) + "是" + self.metion + "的" + self.rel

        elif target_type == "时间":
            question = random.choice(self.time_question_word) + "是" + self.metion + self.rel

        else:
            question = "什么是" + self.metion + "的" + self.rel

        if self.metion and self.rel:
            data = str(
                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question + "\t" + self.metion_type
        return data

    # 头-疑问词-尾
    def type5(self, target_type, n):
        data = ""
        if self.answer_type in self.place:
            question = self.metion + "在" + random.choice(
                self.place_question_word) + self.rel.replace("单位", "") + "的"
        elif self.answer_type in self.person:
            question = self.answer + "是" + random.choice(
                self.people_question_word) + "的" + self.rel
        elif target_type == "时间":
            question = self.metion + "是" + random.choice(self.time_question_word) + self.rel.replace("时间", "").replace(
                "年代", "").replace("日期", "") + "的"
        else:
            question = self.metion + "的什么是" + self.rel

        if self.metion and self.rel and self.answer:
            data = str(
                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question + "\t" + self.metion_type

        return data

    def create_traindata(self, flag):
        data_num = self.datanum
        n = 0
        random.shuffle(self.entity_list)
        if flag == 0:
            if len(self.instance_dict) <= 100000:
                train_num = 50000
            else:
                train_num = 100000
            for n in tqdm(range(train_num)):
                # 随机抽取的三元组
                data = ""
                self.id = random.choice(self.entity_list)
                entity, entity_type, alias, desc = self.instance_dict[self.id]
                if entity_type == "ontology":
                    rel, rel_type, answer_type = random.choice(self.ontology_info[entity])
                else:
                    rel, rel_type, answer_type = random.choice(self.ontology_info[entity_type])
                self.metion = entity
                self.metion_type = entity_type

                self.rel = rel
                self.rel_type = rel_type

                self.answer = ""
                self.answer_type = answer_type
                tagert_type = ""
                if self.answer != "":

                    # 根据targetType 判断答案类型
                    answer_list = jieba.lcut(self.answer)
                    for d in answer_list:
                        if d in self.duliang_danwei:
                            tagert_type = "数字"
                        elif d in self.time_danwei:
                            tagert_type = "时间"
                        else:
                            tagert_type = "其它"

                # 随机选一种要生成的句子类型
                type_chose = random.choice(self.quetion_type)
                # print(type_chose)
                # 生成问句
                '''
                0
                09efdfb0d73cb9353ad6267ea74db6eb
                机械设计基础
                作者
                杨可桢，程光蕴，李仲生    
                《机械设计基础》这本书的作者是谁？
                '''
                question = ""
                # 一些特殊的问句变形
                if "重量" in self.rel:
                    question = self.metion + "有多重"
                elif "大小" in self.rel or "体积" in self.rel or "速度" in self.rel:
                    question = self.metion + "有多大"
                elif "高度" in self.rel:
                    question = self.metion + "有多高"
                elif self.rel == "部署":
                    question = self.metion + "部署了什么"
                elif self.rel == "包含":
                    question = self.metion + "有什么"
                elif self.rel == "研制":
                    question = self.metion + "研制的" + self.answer_type
                else:
                    question = self.metion + "的" + self.rel

                if self.metion and self.rel and self.answer:
                    data = str(
                        n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question + "\t" + self.metion_type
                # print(data)
                if data:
                    self.f_out.write(data + "\n")
                # 2个实体找关系
                # 加农炮和X级潜艇有什么关系
                if type_chose == 0:

                    self.id2 = random.choice(self.entity_list)
                    entity, entity_type, alias, desc = self.instance_dict[self.id]

                    self.metion2 = entity

                    data = self.type1(n)

                elif type_chose == 1:
                    data = self.type2(tagert_type, n)

                elif type_chose == 2:
                    question = self.metion
                    data = str(
                        n) + "\t" + self.id + "\t" + self.metion + "\t" + "null" + "\t" + "null" + "\t" + question + "\t" + self.metion_type

                elif type_chose == 3:
                    data = self.type4(tagert_type, n)
                elif type_chose == 4:
                    data = self.type5(tagert_type, n)

                if data:
                    # print("data++++++++" + str(type_chose) + " " + data)
                    self.f_out.write(data + "\n")
        else:
            for type in self.types:
                # 如果本体类型的实体类型数大于0
                if len(self.ontology_dict[type]) > 1:
                    print(type)
                    for type_num in range(data_num):
                        data = ""
                        self.metion_type = ""
                        # 随机抽取的三元组的实体类型在本体类型列表里的三元组
                        while (self.metion_type == "" or self.metion_type not in self.ontology_dict[
                            type] or self.metion_type != type):
                            self.id = random.choice(self.entity_list)
                            entity, entity_type, alias, desc = self.instance_dict[self.id]
                            rel, rel_type, answer_type = self.ontology_info[entity_type]

                            self.metion = entity
                            self.metion_type = entity_type

                            self.rel = rel
                            self.rel_type = rel_type

                            self.answer = ""
                            self.answer_type = answer_type

                        tagert_type = ""
                        if self.answer != "":
                            # 根据targetType 判断答案类型
                            answer_list = jieba.lcut(self.answer)
                            for d in answer_list:
                                if d in self.duliang_danwei:
                                    tagert_type = "数字"
                                elif d in self.time_danwei:
                                    tagert_type = "时间"
                                else:
                                    tagert_type = "其它"

                        # 随机选一种要生成的句子类型
                        type_chose = random.choice(self.quetion_type)
                        print(type_chose)
                        # 生成问句
                        '''
                        0
                        09efdfb0d73cb9353ad6267ea74db6eb
                        机械设计基础
                        作者
                        杨可桢，程光蕴，李仲生    
                        《机械设计基础》这本书的作者是谁？
                        '''
                        question = ""
                        # 一些特殊的问句变形
                        if "重量" in self.rel:
                            question = self.metion + "有多重"
                        elif "大小" in self.rel or "体积" in self.rel or "速度" in self.rel:
                            question = self.metion + "有多大"
                        elif "高度" in self.rel:
                            question = self.metion + "有多高"
                        elif self.rel == "部署":
                            question = self.metion + "部署了什么"
                        elif self.rel == "包含":
                            question = self.metion + "有什么"
                        elif self.rel == "研制":
                            question = self.metion + "研制的" + self.answer_type
                        else:
                            question = self.metion + "的" + self.rel

                        if self.metion and self.rel and self.answer:
                            data = str(
                                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question + "\t" + self.metion_type
                        print(data)
                        if data:
                            self.f_out.write(data + "\n")
                            n += 1
                        # 2个实体找关系
                        # 加农炮和X级潜艇有什么关系
                        if type_chose == 0:
                            while (self.metion_type == ""
                                   or self.metion_type not in self.ontology_dict[type]
                                   or self.metion_type != type):
                                chosen_two = random.choice(self.lines).strip()
                                # 清洗下三元组
                                self.id2 = random.choice(self.entity_list)
                                entity, entity_type, alias, desc = self.instance_dict[self.id]
                                self.metion = entity

                            data = self.type1(n)

                        elif type_chose == 1:
                            data = self.type2(tagert_type, n)

                        elif type_chose == 2:
                            question = self.metion
                            data = str(
                                n) + "\t" + self.id + "\t" + self.metion + "\t" + "null" + "\t" + "null" + "\t" + question + "\t" + self.metion_type

                        elif type_chose == 3:
                            data = self.type4(tagert_type, n)
                        elif type_chose == 4:
                            data = self.type5(tagert_type, n)

                        if data:
                            print("data++++++++" + str(type_chose) + " " + data)
                            self.f_out.write(data + "\n")
                            n += 1
            self.f_out.close()


def main(kb_file, ontology_type, datapath, train_dataNum, flag):
    # 读文件获取信息
    ontology_dict, instance_dict = read_kb_file(kb_file, datapath)
    # spacy kb 文件
    spacy_path = datapath + "el" + "/assets/"
    if not os.path.exists(spacy_path):
        os.makedirs(spacy_path)
    make_entities(instance_dict, datapath + "el" + "/assets/")
    # 倒排需要的文件
    reach_index(ontology_dict, instance_dict, datapath + "reach.pkl")
    create_inverted_index_entity(instance_dict, datapath + "entity.pkl")
    # 生成训练数据
    gr = train_data_generator(datapath, ontology_type, ontology_dict, instance_dict, train_dataNum)
    gr.create_traindata(flag)


if __name__ == '__main__':
    main("kb_data.json","{}","",100,0)
