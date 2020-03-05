=====================================================
Experiment script for running Document Classification
=====================================================

------------
Introduction
------------
This script executes a classification pipeline for automatically selecting abstracts associated to a specific Software Engineering Systematic Literature Review.
It executes an experiment which has the goal of evaluating how Text Classification techniques can be used to enhance Systematic Literature Reviews selection phase.

This script functionality is five fold:
* It reads and parses BIB file for extracting the title/abstract and whether this study is related to the review or not.
* Feature extraction using TF-IDF.
* Feature selection using Chi-squared distribution to identify most relevant features.
* 5-fold Cross Validation using two classifiers: Decision tree and Support Vector Machines.
* Data report, collecting precision, recall, f-score, and changes in the activation threashold of each classifier.

-------------
Configuration
-------------
This script can be executed using a Docker container.
The proposed Docker container can be build using the presented Dockerfile, or it can be executed using the following public available image: https://hub.docker.com/r/watinha/nltk-keras-gensim

The script dependencies are:
* Numpy
* Sklearn
* NLTK
* matplotlib
* Keras (future studies)

----------
How to run
----------
In order to run the script, the user can execute the file main.py passing as argument which Systematic Literature Review should be used to train and evaluate the classification.
The script was implemented considering the following reviews: games, slr, pair, illiterate, mdwe, testing, ontologies and xbi.

Additionally, there is an all argument which will run the experiment considering all reviews.

The number of features which will be considered in the Data selection phase, should be manually configured within the main.py script.

Furthermore, the results will be stored in CSV files in a result folder.
