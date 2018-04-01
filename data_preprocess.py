import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string=re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ",string)
    string=re.sub(r"\'ve"," \'ve",string)
    string=re.sub(r"n\'t"," n\'t",string)
    string=re.sub(r"\'s",' \'s',string)
    string=re.sub(r"\'re"," \'re",string)
    string=re.sub(r"\'d"," \'d",string)
    string=re.sub(r"\'ll"," \'ll",string)
    string=re.sub(r","," , ",string)
    string=re.sub(r"!"," ! ",string)
    string=re.sub(r"\("," \( ",string)
    string=re.sub(r"\)"," \) ",string)
    string=re.sub(r"\?"," \? ",string)
    string=re.sub(r"\s{2,}"," ",string)
    return string.strip().lower()


def load_data(ps_data,ng_data):
    ps_list=list(open(ps_data,"r").readlines())
    ps_list=[s.strip() for s in ps_list]
    ng_list=list(open(ng_data,"r").readlines())
    ng_list=[s.strip() for s in ng_list]
    review=ps_list+ng_list
    review=[clean_str(s) for s in review]

    ps_label=[[0,1] for _ in ps_list]
    ng_label=[[1,0] for _ in ng_list]
    labels=np.concatenate([ps_label,ng_label],axis=0)
    return [review,labels]


def batch_iter(data,batch_size,epochs,shuffle=True):
    data=np.array(data)
    n_batch=int((len(data)-1)/batch_size) + 1
    for _ in range(epochs):
        if shuffle:
            ind=np.random.permutation(np.arange(len(data)))
            shuffled_data=data[ind]
        else:
            shuffled_data=data
        for i in range(n_batch):
            start=i*batch_size
            end=min((i+1)*batch_size,len(data))
            yield shuffled_data[start:end]
