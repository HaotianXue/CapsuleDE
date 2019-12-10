### 12/07/2019
#### Things have been done:
#### 1. The best classical model except XGB is SVM whose f1_score is about 60. However, it is too time-consuming(4h to train). Compared with other models, I think the most efficient model is Naive Bayes so far.
#### 2. Alothong ML models can do good job, but it seems that DL models outperform most ML models in most cases. I will make attempts to see if they can do better job.
#### 3. It seems like ml models(only with tfidf feature) can only gain f1_score like 50~60 except XGBoost.
#### 4. XGBoost shows unlimited potential in this task, it is still being trained and getting better and better results.According to the results of XGB so far, XGB even outperforms some of DL models. Good job XGB.

#### Things to do next:
#### 1. I will try more rounds for XGB then to see how well it will perform.
#### 2. What about a voting system with XGB and SVM? That is interesting.
#### 3. A totally automatic framework for ML models sounds cool.
#### 4. Time to look into BERT I think.

### 12/07/2019
#### It seems that not all models will perform better with TF-iDF, so I tried all models with both BoW and TF-iDF except Voting. 
#### Since TF-iDF can not promise a better overall performance than BoW, maybe other features like stopwords, stemwrods and lemma will also perform differently in some models. I will find out then.
#### Google Colab can not run a notebook for too long, it is quite annoying. XGBoost can do a better job if rounds are increased to 2k or even 3k sicne so many rounds have been consumed, XGBoost seems still improving, which is good news, however I may now have trouble with running this model.
