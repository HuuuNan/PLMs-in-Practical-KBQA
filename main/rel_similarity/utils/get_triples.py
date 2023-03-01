import json
import os
import re

import tqdm


def get_triple(datapath, kb_data):
    r1 = "[^\u4e00-\u9fa5^a-z^A-Z^0-9]"
    entity_tree = []
    property_tree = []
    relation_tree = []
    ontology_tree = []

    ent2id = {}
    user_dict = []
    name_list = []
    triples_list = []

    entity_set = set()
    relation_set = set()

    for kb_json in kb_data:
        id = kb_json["sourceId"].strip()
        id_target = kb_json["targetId"]
        entity = clean_data(kb_json["sourceName"])
        entity_type = kb_json["sourceType"].strip()
        rel = clean_data(kb_json["relationName"])
        rel_type = kb_json["relationType"].strip()
        if "targetName" in kb_json.keys():
            answer = clean_data(kb_json["targetName"])
        else:
            answer = "null"
        answer = answer.replace("\n", "").replace("\r", "").replace("\t", "")

        if "targetType" in kb_json.keys():
            answer_type = kb_json["targetType"].strip()
            if answer_type != "property":
                name = id_target + "@@@" + answer_type + "@@@" + answer
                name_list.append(name)
                # 实体同义词表
                answer_smi = re.sub(r1, "", answer).lower()
                entity_set.add((answer, answer_smi, answer_type))
        else:
            answer_type = "property"
        # # 实体同义词表
        entity_smi = re.sub(r1, "", entity).lower()
        entity_set.add((entity, entity_smi, entity_type))
        # 三元组
        triples = id + "@@@" + entity + "@@@" + rel + "@@@" + answer + "@@@" + entity_type + "@@@" + rel_type + "@@@" + answer_type
        triples_list.append(triples)
        # names文件
        name = id + "@@@" + entity_type + "@@@" + entity
        name_list.append(name)

        # 关系同义词表
        rel_smi = re.sub(r1, "", rel).lower()
        relation_set.add((rel, rel_smi, rel_type))

        # 构建自动机需要的各类型list
        entity = re.sub(r1, "", entity).lower()
        user_dict.append(entity)
        user_dict.append(rel)

        if entity_type != "ontology":
            entity_tree.append(entity)
        else:
            ontology_tree.append(entity)
        if rel_type != "property":
            relation_tree.append(rel)
        else:
            property_tree.append(rel)

    user_dict = list(set(user_dict))
    name_list = list(set(name_list))
    triples_list = list(set(triples_list))

    entity_tree = list(set(entity_tree))
    ontology_tree = list(set(ontology_tree))
    relation_tree = list(set(relation_tree))
    property_tree = list(set(property_tree))

    # # 实体同义词表，关系同义词表
    entity_list = [entity + '\t' + entity_smi.strip() + '@@@@@' + entity_type.strip() + '\n' for
                   entity, entity_smi, entity_type in list(entity_set)]
    relation_list = [relation_name + '\t' + relation_smi.strip() + '@@@@@' + relation_type.strip() + '\n' for
                     relation_name, relation_smi, relation_type in list(relation_set)]
    entity_list = list(set(entity_list))
    relation_list = list(set(relation_list))

    with open(datapath + "entity_tree.txt", "w", encoding="utf8") as f_1:
        f_1.write("\n".join(entity_tree))
    with open(datapath + "ontology_tree.txt", "w", encoding="utf8") as f_2:
        f_2.write("\n".join(ontology_tree))
    with open(datapath + "relation_tree.txt", "w", encoding="utf8") as f_3:
        f_3.write("\n".join(relation_tree))
    with open(datapath + "property_tree.txt", "w", encoding="utf8") as f_4:
        f_4.write("\n".join(property_tree))

    with open(datapath + "triples.txt", "w", encoding="utf8") as f_o:
        f_o.write("\n".join(triples_list))
    with open(datapath + "user_dict.txt", "w", encoding="utf8") as f_user_dict:
        f_user_dict.write("\n".join(user_dict))
    with open(datapath + "names.txt", "w", encoding="utf8") as f_n:
        f_n.write("\n".join(name_list))

    with open(datapath + 'entity.txt', 'w', encoding='utf-8')as f:
        f.writelines(entity_list)
    with open(datapath + 'relation.txt', 'w', encoding='utf-8')as f1:
        f1.writelines(relation_list)

    return ent2id


def clean_data(mystr):
    mystr = mystr.replace("\r", "", 1000).replace("\r\r", "", 1000).replace("\n", "").replace(".", "-").strip()
    return mystr


if __name__ == '__main__':
    import json

    kb_data = []
    with open("kb_data.json", 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line:
                print(line)
                data = json.loads(line)
                kb_data = kb_data + data
            else:
                break
    print(kb_data)
    get_triple("", kb_data)
