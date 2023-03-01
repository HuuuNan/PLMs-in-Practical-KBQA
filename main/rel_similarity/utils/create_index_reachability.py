#!/usr/bin/python


import pickle


def create_index_reachability(fbpath, outpath):
    print("creating the index map...")
    index = {}
    size = 0
    with open(fbpath, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):

            items = line.strip().split("@@@")

            subject = items[0]
            predicate = items[2]
            object = items[3]
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


if __name__ == '__main__':
    create_index_reachability("triples.txt", "reach.pkl")
    print("Created the reachability index.")
