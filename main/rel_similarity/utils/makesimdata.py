import random
import json


def make_rp_data(rel_path, data_lines, savepath, mode):
    # f = open("train.json", "w", encoding="utf8")
    # f_v = open("valid.json", "w", encoding="utf8")
    # f_t = open("test.json", "w", encoding="utf8")
    f = open(savepath + "/" + mode + ".json", "w", encoding="utf8")
    data = {"questions": [], "golds": [], "negs": []}
    valid = {"questions": [], "golds": [], "negs": []}
    test = {"questions": [], "paths": []}

    # f_data = open(, "r", encoding="utf8")
    # lines = f_data.readlines()
    n = 0
    rel_list = []
    f_Rel = open(rel_path, "r", encoding="utf8")
    rels = f_Rel.readlines()
    for rel in rels:
        rel = rel.strip().split("\t")[0]
        rel_list.append(rel)
    print("长度：",len(data_lines))
    for i in data_lines:
        index = i.split("\t")
        question = index[5]
        gold = index[3]
        print("gold",gold)
        # test["questions"].append(question)
        # test["paths"].append(rel_list)

        if gold == "null":
            pass
        else:
            data["questions"].append(question)
            data["golds"].append(gold)
            negslist = []
            if gold != "关系":
                negslist.append("关系")
            while len(negslist) != 5:

                neg = random.choice(rel_list)
                if neg != gold:
                    negslist.append(neg)
            print(negslist)

            data["negs"].append(negslist)

        # print(data)
        # else:
        #     valid["questions"].append(question)
        #     valid["golds"].append(gold)
        #     negslist = []
        #     while len(negslist) != 5:
        #         neg = random.choice(lines)
        #         if neg != gold:
        #             negslist.append(neg.split("\t")[3])
        #     valid["negs"].append(negslist)
        n = n + 1
    f.write(json.dumps(data, ensure_ascii=False))
    # f_v.write(json.dumps(valid, ensure_ascii=False))
    # f_t.write(json.dumps(test, ensure_ascii=False))
