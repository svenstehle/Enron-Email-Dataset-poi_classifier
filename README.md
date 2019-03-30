# From Udacity: Intro to Machine Learning
Final project on enron case data

### Problem statement:
* Persons Of Interest (POIs) in the enron case need to be identified/predicted 
* Dataset is small with a little over 100 records
* We have an imbalanced classification problem at hand, POIs make up only a small part of the dataset.

### Install and use:
* Clone repo
* Follow final_project/poi_id.py 

### Approach:
* Applied standard techniques including model selection, outlier removals, gridsearch etc. with cross validation
* In the first upload I did not use any oversampling techniques. 
* first upload: 0.5 f1 score

**Update:** 
* tried out oversampling techniques and with borderlineSMOTE I found a gem for this dataset
* second upload: f1 score of 0.977 

### Possible improvements:
* Use the whole Enron Email Corpus as well and incorporate text classification based on email content

### Thoughts and lessons learned:
* Machine learning pipeline with an imbalanced dataset
* Parameter Tuning / GridSearch
* Effect of outlier removals
* Statistical oversampling methods

Really had a lot of fun and learned a lot while working on this dataset! Comments welcome. 
Done with spyder 3.3.1
