#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri


def create_index_reachability(fbpath, outpath):
    print("creating the index map...")
    index = {}
    size = 0
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))

            subject = items[0]
            predicate = items[1]
            object = items[2]
            # print("{}  -   {}".format(subject, predicate))

            if index.get(subject) is not None:
                index[subject].add(predicate)
            else:
                index[subject] = set([predicate])

            size += 1

    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")

print("Freebase subset: {}".format('triple_70M.txt'))
print("Pickle output: {}".format('redict.pkl'))

create_index_reachability('triple_70M.txt', 'redict.pkl')
print("Created the reachability index.")
