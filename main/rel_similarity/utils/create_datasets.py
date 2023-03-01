import json
import random
import jieba
import re

from tqdm import tqdm


class train_data_generator():
    def __init__(self, datapath, ontology_dict, data_num,):

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
        f_Rel = open(datapath + "relation.txt", "r", encoding="utf8")

        # 加载属性关系列表
        self.rel_list = []
        rels = f_Rel.readlines()
        for rel in rels:
            rel = rel.strip().split("\t")[0]
            self.rel_list.append(rel)

        # 添加一些真实的百科简单问句
        f_real = open(datapath + "real_data.txt", "r", encoding="utf8")
        real_data = f_real.readlines()
        for real in real_data:
            real_index = real.strip().split("\t")
            real_type = real_index[3]
            if real_type in self.rel_list:
                self.f_out.write(real)

        # 根据三元组文件来生成训练数据
        f = open(datapath + "triples.txt", "r", encoding="utf8")
        self.lines = f.readlines()

        # 根据类型加疑问词
        self.ontology_dict = ontology_dict
        self.types = ontology_dict.keys()
        # self.place = ontology_dict["地理"] + ontology_dict["军事对象"]
        # self.person = ontology_dict["人"]
        self.place = []
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
                n) + "\t" + self.id + " " + self.id2 + "\t" + self.metion + " " + self.metion2 + "\t" + "关系" + "\t" + "null" + "\t" + question+"\t"+self.metion_type
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
            data = str(n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question+"\t"+self.metion_type
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
                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question+"\t"+self.metion_type
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
                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question+"\t"+self.metion_type

        return data

    def create_traindata(self, flag):
        data_num = self.datanum
        n = 0
        random.shuffle(self.lines)
        if flag == 0:
            if len(self.lines) <= 100000:
                train_num = 50000
            else:
                train_num = 100000
            for n in tqdm(range(train_num)):
                # 随机抽取的三元组
                chosen_one = random.choice(self.lines).strip()
                # 清洗下三元组
                chosen_one = self.cleandata(chosen_one)
                # 按分隔符提取三元组spo
                # 0dbe77aa615dc5d32be54cdce7212436@@@越南陆军@@@百科URL@@@_越南陆军_6548967@@@陆军部队@@@property@@@null
                index = chosen_one.split("@@@")
                print(index)

                self.id = index[0]
                self.metion = ''.join(index[1].split())
                self.metion_type = index[4].strip()
                # print(self.metion_type)
                self.rel = ''.join(index[2].split())
                self.rel_type = index[5]

                self.answer = ''.join(index[3].split())
                self.answer_type = index[6]

                # 根据targetType 判断答案类型
                answer_list = jieba.lcut(self.answer)

                tagert_type = ""
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
                        n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question+"\t"+self.metion_type
                # print(data)
                if data:
                    self.f_out.write(data + "\n")
                # 2个实体找关系
                # 加农炮和X级潜艇有什么关系
                if type_chose == 0:

                    chosen_two = self.cleandata(chosen_one)
                    # 按分隔符提取三元组spo
                    # 0dbe77aa615dc5d32be54cdce7212436@@@越南陆军@@@百科URL@@@_越南陆军_6548967@@@陆军部队@@@property@@@null
                    index = chosen_two.split("@@@")

                    self.id2 = index[0]
                    self.metion2 = ''.join(index[1].split())

                    data = self.type1(n)

                elif type_chose == 1:
                    data = self.type2(tagert_type, n)

                elif type_chose == 2:
                    question = self.metion
                    data = str(
                        n) + "\t" + self.id + "\t" + self.metion + "\t" + "null" + "\t" + "null" + "\t" + question+"\t"+self.metion_type

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
                            chosen_one = random.choice(self.lines).strip()
                            # 清洗下三元组
                            chosen_one = self.cleandata(chosen_one)
                            # 按分隔符提取三元组spo
                            # 0dbe77aa615dc5d32be54cdce7212436@@@越南陆军@@@百科URL@@@_越南陆军_6548967@@@陆军部队@@@property@@@null
                            index = chosen_one.split("@@@")

                            self.id = index[0]
                            self.metion = ''.join(index[1].split())
                            self.metion_type = index[4]
                            print(self.metion_type)
                            self.rel = ''.join(index[2].split())
                            self.rel_type = index[5]

                            self.answer = ''.join(index[3].split())
                            self.answer_type = index[6]
                        print("chosenone", chosen_one)

                        # 根据targetType 判断答案类型
                        answer_list = jieba.lcut(self.answer)

                        tagert_type = ""
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
                                n) + "\t" + self.id + "\t" + self.metion + "\t" + self.rel + "\t" + self.answer + "\t" + question+"\t"+self.metion_type
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
                                chosen_two = self.cleandata(chosen_one)
                                # 按分隔符提取三元组spo
                                # 0dbe77aa615dc5d32be54cdce7212436@@@越南陆军@@@百科URL@@@_越南陆军_6548967@@@陆军部队@@@property@@@null
                                index = chosen_two.split("@@@")

                                self.id2 = index[0]
                                self.metion2 = ''.join(index[1].split())

                            data = self.type1(n)

                        elif type_chose == 1:
                            data = self.type2(tagert_type, n)

                        elif type_chose == 2:
                            question = self.metion
                            data = str(
                                n) + "\t" + self.id + "\t" + self.metion + "\t" + "null" + "\t" + "null" + "\t" + question+"\t"+self.metion_type

                        elif type_chose == 3:
                            data = self.type4(tagert_type, n)
                        elif type_chose == 4:
                            data = self.type5(tagert_type, n)

                        if data:
                            print("data++++++++" + str(type_chose) + " " + data)
                            self.f_out.write(data + "\n")
                            n += 1
            self.f_out.close()


if __name__ == '__main__':

    with open("ontology.txt", "r", encoding="utf8") as f:
        result = f.read()

    result = json.loads(result)
    ontology_dict = {}
    for children in result["data"]["children"]:
        name = children["name"]
        ontology_dict[name] = []
        child = children["children"]
        for ch in child:
            ch_name = ch["name"]
            if ch["children"]:

                for ch_ch in ch["children"]:
                    ch_ch_name = ch_ch["name"]
                    ontology_dict[name].append(ch_ch_name)
            else:
                ontology_dict[name].append(ch_name)
    # print(ontology_dict)
    gr = train_data_generator(datapath="", ontology_dict=ontology_dict, data_num=500)

    gr.create_traindata()
