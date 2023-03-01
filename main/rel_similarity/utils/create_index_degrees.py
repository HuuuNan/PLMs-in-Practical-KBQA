#!/usr/bin/python


import pickle


def create_index_degrees(fbpath, outpath):
    print("creating the index map...")
    index = {}  # indegree, outdegree
    size = 0

    with open(fbpath, 'r',encoding="utf8") as f:

        for i, line in enumerate(f):
            items = line.strip().split("@@@")
            print(items)
            subject = items[0]
            # print(subject)
            predicate = items[2]
            object = items[3]
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


if __name__ == '__main__':
    create_index_degrees("triples.txt", "degree.pkl")
    print("Created the reachability index.")
