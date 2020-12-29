* 이 폴더에 업로드된 스크립트는 DACON에서 개최된 'AI야, 진짜 뉴스를 찾아줘!'에서 제공된 데이터를 바탕으로 분석을 수행함
* 대회 기간: 2020.11.23~2020.12.31

### Current Best Method
* TF-IDF를 인풋으로 활용
* lightGBM의 HPO 필요

### Trial1
* 사용 모델: lightGBM
* Hyperparameter: tokenizer.texts_to_matrix의 mode

#### mode = freq
* Train Accuracy: 96.9%
* Test Accuracy: 95.899%

#### mode = count
* Train Accuracy: 96.4%
* Test Accuracy: 95.678%

#### Conclusion
* Train Accuracy는 freq를 활용할 때 더 높음
* Test Accuracy는 freq를 활용할 때 더 높음
* 따라서, mode는 freq를 사용하는 것이 더 나음

### Trial2
* Trial1과 차이: TDM(기존 최종 입력값)을 TF-IDF로 변환

#### Accuracy
* Train Accuracy: 96.9%
* Test Accuracy: 96.000%


### Trial3
* Naive Bayes 사용

#### Accuracy
* Train Accuracy: 95.8%
* Test Accuracy: 94.505%

### Trial4
* 추가적인 데이터 전처리 수행
* lightGBM HPO

#### Train Accuracy

##### boosting_type

(fixed set: learning rate = 0.1(default), n_estimators = 100(default))

* gbdt(default): 96.826%
* dart: 95.67%
* rf(random forest): fail

##### learning rate & n_estimators
(fixed set: boosting_type = gbdt(best))

* learing rate(0.5-0.1-0.05-0.01): 99.13%-96.826%-95.81%-94.06%
* n_estimators @lr=0.5(50-100-500): 98.28%-99.13%-99.957%