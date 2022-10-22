# Deep Datathon
This project is about identifying the fake websites based on URLs.
# URLs
URL stands for uniform resource locator. It is used to locate a resource such as hypertext pages, images, audio files etc. on web. Base URL is calculated by concatenating protocol, hostname and path of the URL. The generic format of a URL is as follows:

![image](https://user-images.githubusercontent.com/109916989/180658893-4a129e77-5a70-48f0-acc2-f334e22c05d7.png)

## Phishing
Phishing continues to be of major concern not only because of increase in number of phishing attacks but also because of the sophisticated methods used by the attackers to perform the attack.<br>
A Python program has been implemented which takes URL of a website as an input, extracts features from different parts of URL and the features can be either hand-crafted or obtained from TF-IDF. These extracted features are fed to the model trained using machine learning algorithms to classify the website either as legitimate or phishing.

## Importing the packages
Pandas is usually imported under the pd alias<br>
NumPy is a Python package. It stands for 'Numerical Python'. It is a library consisting of multidimensional array objects and a collection of routines for processing of array.<br>
Python Scikit-learn lets users perform various Machine Learning tasks and provides a means to implement Machine Learning in Python.
<br>
### For preprocessing,the techniques used are
- CountVectorizer <br>
It is a great tool provided by the scikit-learn library in Python. It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
- TfidVectorizer <br>
The process of transforming text into a numerical feature is called text vectorization. TF-IDF is one of the most popular text vectorizers, the calculation is very simple and easy to understand. It gives the rare term high weight and gives the common term low weight.

## Machine learning algorithms
To evaluate the performance of CountVectorizer and TF-IDF features, we applied various machine learning classifiers such as XGBoost,  logistic regression, K-Nearest neighbour, MultinomialNB & decision tree to train our proposed model. The main intention of comparing various classifiers is to choose the best classifier suitable for our feature set. To implement various machine learning classifiers, Scikit-learn package is used and Python is used for feature extraction. <br>
### Techniques used for Model Training 
- Logistic Regression() <br>
 Logistic regression is an example of supervised learning. It is used to calculate or predict the probability of a binary (yes/no) event occurring.
- MultinomialNB() <br>
 The Multinomial Naive Bayes algorithm is a Bayesian learning approach popular in Natural Language Processing (NLP). The program guesses the tag of a text, such as an email or a newspaper story, using the Bayes theorem. It calculates each tag's likelihood for a given sample and outputs the tag with the greatest chance.
- XGB Classifier <br>
 XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.
 ## The results are as follows:
 - Logistic Regression() : 92% accurate results<br>
 - MultinomialNB() : 93% accurate results<br>
 - XGB Classifier : 88% accurate results<br>
