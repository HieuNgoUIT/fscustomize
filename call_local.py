import requests

url = 'http://0.0.0.0:3000/'
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/data/hieungotrung/robertadecoder/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 



with open('testset_6000.txt','r') as f1, open('result.txt', 'w') as f:
    for line in f1:
        prep_line = str(" ".join(rdrsegmenter.tokenize(line)[0]))
        #print(prep_line)
        respone = requests.post(url, json = {"text": [prep_line]}).json()
        f.write(respone['summary'][0].strip() + "\n")
        print(respone['summary'][0].strip())

