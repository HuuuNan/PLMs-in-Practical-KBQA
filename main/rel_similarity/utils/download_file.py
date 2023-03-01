import json

import requests

import config.global_var as config


def download_file(file_id_list, dataset_path):
    '''
    '''
    data_list = []
    headers = {'content-type': "application/json",
               'Authorization': 'APP appid = 4abf1a,token = 9480295ab2e2eddb8'}

    for file_id in file_id_list:
        r = requests.get(config.NLP_TASK_IP + '/api/v1/platform/file/one' + '?fileId=' + str(file_id),
                         headers=headers, timeout=36000)

        data = json.loads("".join(
            str(r.content, encoding="utf-8").replace('\\n', '', 1000000000000).replace('\t', '', 1000000000000).replace('\n\n', '',
                                                                                                 1000000000000).replace(
                '\n', '', 1000000000000).split()))
        data_list += data

    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False)
    return dataset_path, data_list


def download_ontology_file(ontologyId, dataset_path):
    '''
    '''
    data_list = []
    headers = {'content-type': "application/json",
               'Authorization': 'APP appid = 4abf1a,token = 9480295ab2e2eddb8'}
    data = {"treeMode":True}
    jsondata = json.dumps(data)

    r = requests.post(config.NLP_TASK_IP + '/api/v1/platform/ontology/' + ontologyId + '/object/query', headers=headers, timeout=36000,data=jsondata)
    print(r.text)

    data = json.loads("".join(
        str(r.content, encoding="utf-8").replace('\\n', '', 1000000000000).replace('\t', '', 1000000000000).replace('\n\n', '',
                                                                                             1000000000000).replace(
            '\n', '', 1000000000000).split()))
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return dataset_path, data_list


if __name__ == '__main__':
    download_ontology_file('8e66b792290642749a86c002a98cbca9','ontology_data.txt')
