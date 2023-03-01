#!/usr/bin/python

import sys
import argparse
import pickle

from util import www2fb, clean_uri


def create_index_degrees(fbpath, outpath):
    print("creating the index map...")
    index = {} # indegree, outdegree
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

            # increment outdegree of subject
            if index.get(subject) is not None:
                index[subject][1] += 1
            else:
                index[subject] = [0, 1]
            # increment the count of predicate - first index, 2nd index is useless here
            if index.get(predicate) is not None:
                index[predicate][0] += 1
            else:
                index[predicate] = [1, 0]
            # increment the indegree of object
            if index.get(object) is not None:
                index[object][0] += 1
            else:
                index[object] = [1, 0]

            size += 1

    print("num keys: {}".format(len(index)))
    print("total key-value pairs: {}".format(size))

    with open(outpath, 'wb') as f:
        pickle.dump(index, f)

    print("DONE")

print("Freebase subset: {}".format('triple_70M.txt'))
print("Pickle output: {}".format('degree.pkl'))

create_index_degrees('triple_70M.txt', 'degree.pkl')
print("Created the reachability index.")
