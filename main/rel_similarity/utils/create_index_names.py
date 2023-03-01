#!/usr/bin/python

import pickle

def get_names_for_entities(namespath,pklpath):
    print("getting names map...")
    names = {}
    with open(namespath, 'r',encoding="utf8") as f:
        for i, line in enumerate(f):
            items = line.strip().split("@@@")
            id = items[0]
            entity = items[2]

            if entity != "":
                if names.get(id) is None:
                    names[id] = [(entity)]
                else:
                    names[id].append(entity)
    with open(pklpath, 'wb') as f:
        pickle.dump(names, f)
    # return names

if __name__ == '__main__':

    index_names = get_names_for_entities("names.txt")
    print(index_names)



    print("Created the names index.")
