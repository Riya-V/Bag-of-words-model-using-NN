#train bow using linear regression
#tqdm,pickle,pca,
#creating BoW model
import pandas as pd 

file=open('cooking.stackexchange.txt','r')
#readlines-Reads all the lines and return them as each line a string element in a list.
text=file.readlines()  #each line in file will be stored in list(named text) as string
file.close()#close the file(good practise)


#print(text) prints list

sentences=[]
label = []
l_target = []
for r in text:
    list1 = []
    f = r.strip().split()
    for i in f:
        if i[0:9] == '__label__':
            label.append(i[9:])
            list1.append(i[9:])#break
    n = r.strip().split("__label__") #sentence
    m = n[-1].split()
    k=m[0]   
    s = ' '.join(i for i in m[1:])
    
    sentences.append(s)
    l_target.append(list1)

#print(labels[0:6])
#print(sentences[0:6])
#len(labels)-15404
#print(len(sentences))

import pandas as pd
import numpy as np

label=np.array(label)
sentences=np.array(sentences)

#print(len(np.unique(label)))#-736
#exit(0) 

import string
import nltk
from nltk.corpus import stopwords
#nltk.download()-already done
#remove punctuations from string
def text_process(mess):
    no_punct=[char for char in mess if char not in string.punctuation]
    no_punct=''.join(no_punct)
    s= ' '.join(word.lower() for word in no_punct.split() if word.lower() not in stopwords.words('english'))
    
    return s
    #returns list of words in sentences
#print(text_process('How much does potato starch affect a cheese sauce recipe?'))
for idx,i in enumerate(sentences):
    sentences[idx]=text_process(i)
    
 
wordfreq={}

for sent in sentences:
    
    token=nltk.word_tokenize(sent)
    for t in token:
        if t in wordfreq.keys():
            wordfreq[t]+=1
        else:
            wordfreq[t]=0

#for idx,key in enumerate(np.unique(labels)):
#labelmap[idx]=key


#print(len(label))#-35542
#print(len(wordfreq.keys()))#-9470

#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(sentences)
#print(vectorizer.get_feature_names())
#['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this','apple'] 1 1 1
#bowvec=(X.toarray())
#print(bowvec[0])
#exit(0)
from tqdm import tqdm
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
onehot_target=np.zeros( (len(sentences),len(np.unique(label))) )

labelmap={i:no for no,i in enumerate(np.unique(label))}
#onehot_target[range(len(labels)),labelmap.values()]=1


 
for i in range(len(l_target)):
    for l in l_target[i]:
        
        onehot_target[i,labelmap[l]] = 1
#Y_train = y_data
#print(labelmap)
#print(onehot_target[0])
#print(onehot_target[0].shape)
#print(bowvec.shape)
#print(onehot_target.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(bowvec,onehot_target,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
# fitting
scalar.fit(x_train) #use same scaler for train and test
#scalar.fit(x_test)
scaled_data = scalar.transform(x_train)
# Importing PCA
pca = PCA(n_components = 1000)
pca.fit(scaled_data)
x_train_pca = pca.transform(scaled_data)
scaled_data = scalar.transform(x_test)
x_test_pca = pca.transform(scaled_data)

    

from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
model=LinearRegression()
model.fit(x_train_pca,y_train)
pred=model.predict(x_test_pca)
pred_i=[np.argmax(pred[i]) for i in range(len(pred))]

#y_test_i=[np.argmax(y_test_i[i]) for i in range(len(pred))]
#pred=pred
#exit(0)
#print('CEL for LinearRegression:',CEL(y_test,pred))
#print()
import numpy as np
def accuracy(y_test,pred_i):
    correct = 0
    for i in range(len(pred_i)):
        if y_test[i][pred_i[i]]==1:
            correct+=1
    return (correct/len(pred))*100

   
print('Accuracy for LinearRegression:',accuracy(y_test,pred_i))