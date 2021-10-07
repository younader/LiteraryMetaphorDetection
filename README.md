# LiteraryMetaphorDetection
This project explores literary metaphor detection (Binary classification) and user rating prediction (10 regression targets)
Literary metaphors can be detected on the basis of the 10 user ratings in the katz. et al dataset, . The classification section of this projects explores reproducing similar results using automated feature pipelines, both on the word level and on the document level. 
| Method                           | ROC AUC      | # features    |
| -------------------------------- | -------- |-------------|
| Logistic Regression +Transformer Ensembled Embedding              | .89| 4070        |
| GPT2--medium-finetuned                   | .85| 1080    |
| Flair GPT2 Large Classifier  | .97 | 1280     |
| Logistic Regression + engineered features | .75  | 14    |

