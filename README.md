# Definition_Extraction

## Authors: Haotian Xue, Zirong Chen(github: RexZChen)

SemEval2020 Task6 Subtask 1

Task: Given a sentence, tell if it is definitional.

Dataset: SemEval2020 Task6 Dataset

Results of all models: https://docs.google.com/spreadsheets/d/1vgcH6rOkrOY9ApE03D2DbhO_He2Yru2v0DRnnth6xrc/edit#gid=0

Best Model(so far):

DL: Bert(fine-tunned) + Capsule F1-score: 79.97%

ML: XGBoost F1-score: 64.689%
Feature Selection:
1. BoW vs TF-iDF
![GitHub Logo](/img/BoW_&_TF-iDF.png)

Here is the image of AUC's of both train and test sets while rounds goes up.

![GitHub Logo](/img/XGBoost.png)
Following is the tree formed during XGBoosting.
![GitHub Logo](/img/1400_tfidf_tree.png)





