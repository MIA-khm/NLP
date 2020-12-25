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


# 2. 특수문자, 숫자, 영어 삭제
del_words = [',','.','-','+','~','"','·','&','(',')','[',']','=','%','/', #특수문자
             '0','1','2','3','4','5','6','7','8','9', #숫자
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] #영어

train['content_remove'] = train['content']
for del_word in del_words:
    train['content_remove'] = train.content_remove.apply(lambda x: str(x).replace(del_word, ''))


# 3. 토큰화
from konlpy.tag import Okt

word_list = ['스탁론', '무단전재', '재배포', '연계', '동행세일', '비주얼', '역동적인', '카톡방', '신고가',
             '급등', '뉴스봇', '보유세','재추진'] #끊어지는 데이터 목록

okt = Okt()

X_train = []

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

# 4. Make Document-Term matrix(train x set) & train y set

from tensorflow.keras.preprocessing.text import Tokenizer

setted_mode = 'freq' # mode=freq or mode=count

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
token_idx = tokenizer.word_index
x_train = tokenizer.texts_to_matrix(X_train, mode=setted_mode) 
y_train = train['info']

# 5. Train classification model: lightGBM

from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(x_train, y_train)

train_pred = lgb.predict(x_train)

# 6. Check the accuracy
rlt = pd.DataFrame({'real_info':y_train,'pred_info':train_pred})
acc = sum(abs(rlt['real_info'] - rlt['pred_info'])) / np.shape(rlt)[0]
print(acc)

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
test_pred = lgb.predict(x_test)

# get submission

for i in range(np.shape(submission)[0]):
    submission['info'][i] = test_pred[i]
    
submission.to_csv(f'trial1_{setted_mode}.csv', index = False)