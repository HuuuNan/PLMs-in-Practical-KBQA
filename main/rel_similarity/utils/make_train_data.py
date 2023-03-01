import random
import json

from models.el.el_predict import entity_linking


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

    # print("长度：", len(data_lines))
    for i in data_lines:
        index = i.split("\t")
        if len(index) == 7:
            name = index[2]
            question = index[5]
            gold = index[3]
            # print("gold", gold)
            # test["questions"].append(question)
            # test["paths"].append(rel_list)

            if gold == "null":
                pass
            if gold == "关系":
                gold_guanxi = random.choice([0, 1, 2])
                if gold_guanxi == "1":
                    gold == "关系"
                else:
                    pass
            else:
                data["questions"].append(question)
                data["golds"].append(gold)
                negslist = []
                if gold != "关系":
                    negslist.append("关系")
                # mids = entity_linking(name, 1)
                # mid_rel_list = []
                # if mids:
                #     for (mid, mid_name, mid_type), mid_score in mids[0]:
                #         if mid in index_reach.keys():
                #             for rel in list(index_reach[mid]):
                #                 mid_rel_list.append(rel)

                while len(negslist) != 5:
                    # if mid_rel_list:
                    #     neg = random.choice(mid_rel_list)
                    # else:
                    neg = random.choice(rel_list)
                    if neg != gold:
                        negslist.append(neg)
                # print(negslist)

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


def make_ner_data(data, savepath, mode):
    n = 0
    f = open(savepath + "/" + mode + ".json", "w", encoding="utf8")
    for i in data:
        # print(i)
        # f = open(savepath + "/" + str(n) + ".json", "w", encoding="utf8")

        train_data = {
            "content": "",
            "object": []
        }

        n += 1

        temp = i.strip().split("\t")
        if len(temp) == 7:
            text = temp[5]
            # print(text)
            train_data["content"] = text
            mention_data = temp[2]
            mention_data = mention_data.split(" ")
            for j in range(len(mention_data)):
                e_name = mention_data[j]
                # print(mention_data[j])
                if e_name in text:
                    start = text.index(e_name)
                    end = start + len(e_name)

                    label = "entity"

                    info = {"span": e_name, "label": label, "start": start, "end": end}

                    train_data["object"].append(info)

            f.write(json.dumps(train_data, ensure_ascii=False) + "\n")


if __name__ == '__main__':

    f = open("train.json", "r", encoding="utf8")
    f_t = open("../data/train.json", "w", encoding="utf8")
    f_v = open("../data/valid.json", "w", encoding="utf8")
    # f_te = open("../data/test.txt", "w", encoding="utf8")

    lines = f.readlines()
    random.shuffle(lines)
    train_list = []
    test_list = []
    valid_list = []
    for i in range(len(lines)):
        if i < int(len(lines) * 0.8):
            train_list.append(lines[i])

            f_t.write(lines[i])
        else:
            test_list.append(lines[i])
            f_v.write(lines[i])
        # if i < int(len(lines) * 0.8) and i > int(len(lines) * 0.6):
        #     valid_list.append(lines[i])
        #     f_v.write(lines[i])
        # if i > int(len(lines) * 0.2):
        #     test_list.append(lines[i])
        #     f_te.write(lines[i])

    # make_ner_data(train_list, "ner/train")
    # # make_ner_data(valid_list, "ner/valid")
    # make_ner_data(test_list, "ner/test")
