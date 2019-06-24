# What is Machine Learning?
Machine learning (or ML for short) is the process by which a computer program is able to progressively improve performance on a task through *training* and data without explicitly being programmed. In other words, it is a way of programming with data. After an estimator has been trained, we can use it to make predictions about future outcomes which we refer to as *inference*. There are two types of machine learning that Rubix supports out of the box - Supervised and Unsupervised.

## Supervised
Supervised learning is a type of machine learning that includes a training signal in the form of human annotations called *labels* along with the training samples. The training signal guides the learner to output predictions that resemble values found in the labels of the training set.

### Classification
For classification problems, a learner is trained to differentiate samples among a set of K possible discrete classes. In this type of problem, the labels provided are the classes that each sample belongs to such as *"cat"*, *"dog"*, or *"ship"*. Examples of classification problems include image recognition, text sentiment analysis, and Iris flower species classification.

> **Note**: As a convention throughout Rubix ML, discrete variables are always denoted by string type.

### Regression
Unlike classification, where the number of possible class predictions an estimator can make is 1 of K, a Regressor can predict an infinite range of continuous values. In this case, the labels are the desired output values of the estimator given each training sample as input. Regression problems include determining the angle of an automobile steering wheel, estimating the sale price of a home, and credit scoring.

## Unsupervised
A form of learning that does not require training labels is referred to as Unsupervised learning. Instead, Unsupervised learners aim to detect patterns in raw data. Since it is not always easy or possible to obtain labeled data, an Unsupervised learning method is often the first step in discovering knowledge about your data.

### Clustering
Clustering takes a dataset of unlabeled samples and assigns each a discrete label based on their similarity to each other. Example of where clustering is used is in tissue differentiation in PET scan images, customer database market segmentation, or to discover communities within social networks.

### Anomaly Detection
Anomaly detection is the flagging and/or ranking of samples within a dataset based on how different or rare they are. Anomaly detection is commoonly used in security for intrusion and denial of service detection, and in the financial industry to detect fraud.

### Manifold Learning
Manifold learning is a type of non-linear dimensionality reduction used for densely embedding feature representations such as for visualizing high dimensional datasets in low (1 to 3) dimensions and modelling language as word vectors.