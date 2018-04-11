import json
import codecs
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle
import re
import unicodedata
import sys
from sklearn import metrics

is_cuda = False

if torch.cuda.is_available():
    is_cuda = True


def getclassmore(a):
    if a>3: return a-2
    elif a<3: return 0
    else: return 1

def getclass(a):
    if a>3:
        return 2
    elif a<3:
        return 0
    else:
        return 1

def allupper(word):
    for c in word:
        if not(c.isupper()):
            return False
    return True

class Model(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, dict_size, max_sent) :
        super(Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.max_sent = max_sent
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.linearOut = nn.Linear(2*hidden_dim,4)
    def forward(self, inputs) :
        global is_cuda
        if is_cuda:
            hidden = (Variable(torch.zeros(2, len(inputs), self.hidden_dim).cuda()),Variable(torch.zeros(2, len(inputs), self.hidden_dim).cuda()))
        else:
            hidden = (Variable(torch.zeros(2, len(inputs), self.hidden_dim)),Variable(torch.zeros(2, len(inputs), self.hidden_dim)))
        x = self.embeddings(inputs).view(self.max_sent, len(inputs), -1)
        lstm_out,lstm_h = self.lstm(x,hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = torch.sum(x, dim = 0, keepdim = True)
        x = F.log_softmax(x)
        return x

print("Reading Test Data")

test_data = []
test_it = 0
with codecs.open(sys.argv[1],'rU','utf-8') as f:
    for line in f:
        test_it = test_it + 1
        test_data.append(json.loads(line))

test_sentences = []
i = 0
start = test_it
while(i<start):
    sent = test_data[i]['reviewText']
    temp = ""
    for j in range(0, 3):
        temp = temp + ". " + test_data[i]['summary'] + "."
    sent = sent + temp
    test_sentences.append(sent)
    i = i+1

max_sent = 25

limit = 500000

my_dicts = []
my_models = []

num_models = 9

print("Reading models")
for i in range(0, num_models):
    temp = Model(50, 100, 500001, max_sent)
    if is_cuda:
        temp.load_state_dict(torch.load('ensemble/part' + str(i+1) + '.pth'))
    else:
        temp.load_state_dict(torch.load('ensemble/part' + str(i+1) + '.pth', map_location=lambda storage, loc: storage))
    my_models.append(temp)
    if is_cuda:
        my_models[i] = my_models[i].cuda()

for i in range(0, num_models):
    with open('ensemble/dict_part' + str(i+1) + '.pkl', 'rb') as f:
        temp_dict = pickle.load(f)
    my_dicts.append(temp_dict)

my_lab = []

print("Predicting")
print("Will take time. Please read writeup.txt")

for j in range(0, len(test_sentences)):
    if(j%100==0):
        print("Predictions done :", j)
    sent = test_sentences[j]
    fnl_inp = []
    temp_arr = [0.0, 0.0, 0.0, 0.0]
    my_lab.append(2)
    count0 = 0
    count1 = 0
    for i in range(0, num_models):
        input_data = []
        for sng_sen in sent.split('.'):
            if(len(sng_sen.split())==0):
                continue
            if(len(sng_sen.split())>max_sent):
                sng_sen = sng_sen[:max_sent]
            while(len(sng_sen.split())<max_sent):
                sng_sen = sng_sen + " UNK"
            temp_data = []
            for word in sng_sen.split():
                if word in my_dicts[i]:
                    temp_data.append(my_dicts[i][word])
                else:
                    temp_data.append(0)
            input_data.append(temp_data)
        if is_cuda:
            inp_data = Variable(torch.LongTensor(input_data).cuda())
        else:
            inp_data = Variable(torch.LongTensor(input_data))
        y_pred = my_models[i](inp_data)
        lm = y_pred.data.cpu().numpy()
        t = np.argmax(lm)
        if t==0:
            count0 += 1
            if count0>2:
                my_lab[j] = t
                break
        if t==1:
            count1 += 1
            if count1>2:
                my_lab[j] = t
                break
        # temp_arr = temp_arr + lm
        # fnl_inp.append(temp_arr)
    # t = np.argmax(fnl_inp)
    # if t==3:
        # t=2

file = open(sys.argv[2],'w')
for ele in my_lab:
    if(ele==0):
        file.write("1\n")
    elif(ele==1):
        file.write("3\n")
    elif(ele==2):
        file.write("5\n")