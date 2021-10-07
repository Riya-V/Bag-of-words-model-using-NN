import nltk
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
file = open('cooking.stackexchange.txt', 'r')
# read all text
text = file.readlines()
# close the file
file.close()
sentences=[]
labels=[]
from tqdm import tqdm
for i in tqdm(text):
    p=i.strip().split("__label__")
    p.pop(0)
    k=p.pop()
    k=k.split(" ")
    m=k.pop(0)
    p.append(m)
    s=''
    for i in range(len(p)):
        p[i] = p[i].strip()
    sentences.append(s.join([i.lower()+" " for i in k]))
    labels.append(p)
labels=np.array(labels)
sentences=np.array(sentences)
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
import string
for i in tqdm(range(len(sentences))):
    sentences[i]=' '.join(text_process(sentences[i]))
idx=0
d={}
idx=0
for j in labels:
    for i in j:
        if i not in d.keys():
            d[i]=idx
            idx+=1
        else:
            continue
print(len(d.keys()))
wordfreq={}
for sent in sentences:
    token=nltk.word_tokenize(sent)
    for t in token:
        if t in wordfreq.keys():
            wordfreq[t]+=1
        else:
            wordfreq[t]=0
print(len(wordfreq.keys()))
bowvec=[]
for sent in tqdm(sentences):
    token=nltk.word_tokenize(sent)
    sentbowvec=[]
    for t in list(wordfreq.keys()):
        if t in token:
            sentbowvec.append(token.count(t))
        else:
            sentbowvec.append(0)
    bowvec.append(sentbowvec)
bowvec=np.array(bowvec)
from sklearn import preprocessing
bowvec=preprocessing.normalize(bowvec)
bowvec=bowvec.astype(np.float32)
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
pca=PCA(n_components=900)
pca.fit(bowvec)
bowvec=pca.transform(bowvec)
lab_onehot=np.zeros([len(labels),1266])
for i in tqdm(range(len(labels))):
    for k in labels[i]:
        lab_onehot[i][d[k]]=1
sent_train,sent_test,label_train,label_test=train_test_split(bowvec,lab_onehot,test_size=0.2)


class customdataloader(torch.utils.data.Dataset):
    def __init__(self, sent, lab, d):
        self.sent = sent
        self.lab = lab
        self.d = d

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        label = self.lab[idx]
        sentence = self.sent[idx]

        return sentence, label
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(900,100)
        self.fc2=nn.Linear(100,1266)
    def forward(self,x):
        l1=self.fc1(x)
        al1=F.relu(l1)
        l2=self.fc2(al1)
        return l2
def train(model,use_cuda,train_loader,optimizer,epoch):
    model.train()
    for batchid,(data,target) in enumerate(train_loader):
        y_onehot=target.argmax(dim=1,keepdim=True)
        y_onehot=torch.flatten(y_onehot)
        if use_cuda:
            data,y_onehot=data.cuda(),y_onehot.cuda()
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,y_onehot)
        loss.backward()
        optimizer.step()
        if batchid % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batchid * len(data), len(train_loader.dataset),
            100. * batchid / len(train_loader), loss.item()))
def test(model,use_cuda,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            y_onehot=target
            if use_cuda:
                data,y_onehot=data.cuda(),y_onehot.cuda()
            output=model(data)
            #test_loss+=torch.sum((output-y_onehot)**2)
            pred=output.argmax(dim=1,keepdim=True)
            for i in range(len(pred)):
                if target[i][pred[i]]==1:
                    correct+=1
        print(correct)
        #test_loss/=len(test_loader.dataset)
        print(100*correct/len(test_loader.dataset))
def seed(seed_value):
    torch.cuda.manual_seed_all(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
def main():
    use_cuda=False
    seed(0)
    data1=customdataloader(bowvec,lab_onehot,d)
    data2=customdataloader(sent_test,label_test,d)
    train_loader=torch.utils.data.DataLoader(data1,num_workers=0,batch_size=30,shuffle=True)
    test_loader=torch.utils.data.DataLoader(data2,num_workers=0,batch_size=50,shuffle=False)
    model=Net()
    if use_cuda:
        model=model.cuda()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(1,11):
        train(model,use_cuda,train_loader,optimizer,epoch)
        test(model,use_cuda,test_loader)
import random
if __name__=="__main__":
    main()
#1)accuracy after training for 10 epochs-> <b>69.06848425835767<b>


#2)accuracy by changing loss function from cross entropy to categorical cross entropy(CCE) and keeping all other same-><b>69.68516715352159<b<
# cce_loss = torch.nn.CrossEntropyLoss()
#loss = cce_loss( output,y_onehot)
#loss = np.sum(-y_onehot * np.log(output) - (1 - y_onehot) * np.log(1 - output)) [line 169,170 dooes the work which is done by line 171]
#by changing lr from 0.01 to 0.012 in this case <b>accuracy->72.02207075624797<b>


#3)By repacing sigmoid activation function with softmax and changing loss function from crossentropy to CCE(categorical cross entropy) from above code
#accucarcy was decreased to 16.58552418046089%
#import torch.nn as nn
#softmax = nn.Softmax(dim=-1)
#y = softmax(l1)
        
