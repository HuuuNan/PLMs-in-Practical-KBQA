import random

import json


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

        f.write(json.dumps(train_data, ensure_ascii=False)+"\n")


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
