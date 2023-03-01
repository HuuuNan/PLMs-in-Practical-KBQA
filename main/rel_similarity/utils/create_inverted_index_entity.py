#!/usr/bin/python
import jieba

import pickle
import re
# jieba.load_userdict("../data/user_dict.txt")

def get_all_ngrams(tokens):

    all_ngrams = set()
    # max_n = min(len(tokens), 3)
    max_n=len(tokens)
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
    name_tokens=list(jieba.cut(processed_name))
    # print(name_tokens)
    name_ngrams = get_all_ngrams(name_tokens)

    return name_ngrams


def create_inverted_index_entity(namespath, outpath):
    print("creating the index map...")
    index = {}
    size = 0
    with open(namespath, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            items = line.strip().split("@@@")

            entity_mid = items[0]
            entity_type = items[1]
            entity_name = items[2]
            # r1 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘'《》【】￥……（）]+"
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


if __name__ == '__main__':
    r1 = "[^\u4e00-\u9fa5^a-z^A-Z^0-9]"
    print(re.sub(r1, "", "F！22攻击机"))
    # text = re.sub(r1, "", )
    # text = ("F-22").strip(r1).lower()
    # print(text)
    create_inverted_index_entity("names.txt", "entity.pkl")
    print("Created the entity index.")
