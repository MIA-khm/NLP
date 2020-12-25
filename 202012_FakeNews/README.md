* 이 폴더에 업로드된 스크립트는 DAICON에서 개최된 'AI야, 진짜 뉴스를 찾아줘!'에서 제공된 데이터를 바탕으로 분석을 수행함
* 대회 기간: 2020.11.23~2020.12.31

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