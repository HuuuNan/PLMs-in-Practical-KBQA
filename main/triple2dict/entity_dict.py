#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri, processed_text

def get_all_ngrams(tokens):
    all_ngrams = set()
    max_n = min(len(tokens), 3)
    for n in range(1, max_n+1):
        ngrams = find_ngrams(tokens, n)
        all_ngrams = all_ngrams | ngrams
    return all_ngrams

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)


def get_name_ngrams(entity_name):
    processed_name = processed_text(entity_name) # lowercase the name
    name_tokens = processed_name.split()
    name_ngrams = get_all_ngrams(name_tokens)

    return name_ngrams


def create_inverted_index_entity(namespath, outpath):
    print("creating the index map...")
    num=0
    index = {}
    size = 0
    names=pickle.load(open('names.pkl','rb'))
    for i in names.items():
        num+=1
        if num%10000==0:
            print(num)
            print(i)
        for j in i[1]:

            entity_mid = i[0]
            entity_name = j

            name_ngrams = get_name_ngrams(entity_name)

            for ngram_tuple in name_ngrams:
                size += 1
                ngram = " ".join(ngram_tuple)
                # print(ngram)
                if index.get(ngram) is not None:
                    index[ngram].add((entity_mid, entity_name))
                else:
                    index[ngram] = set([(entity_mid, entity_name)])


    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    print("dumping to pickle...")
    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")

create_inverted_index_entity('names.pkl', 'entity.pkl')
print("Created the entity index.")
