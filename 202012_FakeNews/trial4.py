# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:41:38 2020

@author: USER
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/USER/Desktop/news/data')

# 1. Load Data

train = pd.read_csv('news_train.csv')
test = pd.read_csv('news_Test.csv')
submission = pd.read_csv('sample_submission.csv')
stopwords = sum(pd.read_csv('stopwords.csv', encoding ="CP949").values.tolist(),[])
word_list = sum(pd.read_csv('word_list.csv', encoding='CP949').values.tolist(),[]) #끊어지는 데이터 목록



# 2. 특수문자, 숫자, 영어 삭제
del_words = [',','.','-','+','~','"','·','&','(',')','[',']','=','%','/','_','>','<', #특수문자
             '0','1','2','3','4','5','6','7','8','9',] #숫자
#             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', #영어 소문자
#             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] #영어 대문자

train['content_remove'] = train['content']
for del_word in del_words:
    train['content_remove'] = train.content_remove.apply(lambda x: str(x).replace(del_word, ''))


# 3. 토큰화
from konlpy.tag import Okt

okt = Okt()

X_train = []
X_train_arr = []

for sentence in train['content_remove']:
    
    tmp_X = []
    
    for target_word in word_list:
        c = sentence.count(target_word)
        if c > 0:
            for _ in range(c):   
                tmp_X.append(target_word)
        sentence = sentence.replace(target_word,'')

    tmp_X += okt.morphs(sentence, stem=True)
    tmp_X = [word for word in tmp_X if not word in stopwords] # 불용어 제거
    X_train.append(tmp_X)
    X_train_arr.append(np.asarray(tmp_X))


# 4. Make Document-Term matrix(train x set) & train y set

from tensorflow.keras.preprocessing.text import Tokenizer

setted_mode = 'count' # mode=freq or mode=count

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
token_idx = tokenizer.word_index
x_train = tokenizer.texts_to_matrix(X_train, mode=setted_mode) 
y_train = train['info']


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

x_train = tfidf_transformer.fit_transform(x_train)



# 5. Train classification model: lightGBM
#from sklearn.model_selection import train_test_split
#tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(x_train, y_train, test_size=0.3, shuffle = True, random_state=123)


from lightgbm import LGBMClassifier

lgb = LGBMClassifier(boosting_type='gbdt', #gbdt, dart, goss
                     objective = 'binary',
                     learning_rate = 0.1,
                     feature_fraction = 0.8,
                     reg_alpha=0.1,
                     reg_lambda=0.1,
                     n_estimators=5000)
lgb.fit(x_train, y_train)
#evals = [(tmp_x_test, tmp_y_test)]
#lgb.fit(tmp_x_train, tmp_y_train, early_stopping_rounds=100, eval_metric = 'logloss', eval_set=evals, verbose=True)

train_pred = lgb.predict(x_train)
#train_pred = lgb.predict(tmp_x_train)
#test_pred = lgb.predict(tmp_x_test)

# 6. Check the accuracy
rlt = pd.DataFrame({'real_info':y_train,'pred_info':train_pred})
#rlt = pd.DataFrame({'real_info':tmp_y_train,'pred_info':train_pred})

acc = sum(abs(rlt['real_info'] - rlt['pred_info'])) / np.shape(rlt)[0]
print(f'train_acc: {acc}')

#rlt = pd.DataFrame({'real_info':tmp_y_test,'pred_info':test_pred})
#acc = sum(abs(rlt['real_info'] - rlt['pred_info'])) / np.shape(rlt)[0]
#print(f'test_acc: {acc}')

# Test #######################################################################

test['content_remove'] = test['content']
for del_word in del_words:
    test['content_remove'] = test.content_remove.apply(lambda x: str(x).replace(del_word, ''))
    
X_test = []

for sentence in test['content_remove']:
    
    tmp_X = []
    
    for target_word in word_list:
        c = sentence.count(target_word)
        if c > 0:
            for _ in range(c):   
                tmp_X.append(target_word)
        sentence = sentence.replace(target_word,'')

    tmp_X += okt.morphs(sentence, stem=True)
    tmp_X = [word for word in tmp_X if not word in stopwords] # 불용어 제거
    X_test.append(tmp_X)

x_test = tokenizer.texts_to_matrix(X_test, mode=setted_mode)
x_test = tfidf_transformer.transform(x_test)
test_pred = lgb.predict(x_test)

# get submission

for i in range(np.shape(submission)[0]):
    submission['info'][i] = test_pred[i]
    
submission.to_csv('trial4.csv', index = False)


# checking
#tmp = np.asarray(X_train)
#false_data = tmp[y_train==1]
#false_data = pd.DataFrame(false_data)
#false_data.to_csv('false_data.csv', index = False,encoding='CP949')
