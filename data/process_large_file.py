import json

file_path = './moss-003-sft-data.jsonl'
moss_test = '/home/sai/Downloads/moss_test.jsonl'
xingfan_path = '/home/sai/Documents/Firefly/data/xingfan.jsonl'


with open(file_path, 'r') as f:
    mil = 0
    two_mil = 0
    four_mil = 0
    m = 0
    for line in f:
        length = 0
        data_dict = json.loads(line)['conversation']
        for t in data_dict:
            length += len(t['human'])
            length += len(t['assistant'])
        if length < 1024:
            m += 1

        if length > 1024:
            mil += 1
            #print('800:', line)
        if length > 2048:
            two_mil += 1
        if length > 4096:
            four_mil += 1
    print(m, mil, two_mil, four_mil)


