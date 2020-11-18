# Data_Camp_Data_Science
This repo contains detailed data analysis on introduction to statistics, introduction to probability, supervised learning with scikit-learn and unsupervised learning with scikit learn.

## Statistics
Statistics is the study of how to collect, analyze, and draw conclusions from data. It’s a hugely valuable tool that you can use to bring the future into focus and infer the answer to tons of questions. For example, what is the likelihood of someone purchasing your product, how many calls will your support team receive, and how many jeans sizes should you manufacture to fit 95% of the population? In this repo, most situations like these where covered, using scatterplots to show the relationship between numeric values, and calculating correlation. Probability was also tackled, the backbone of statistical reasoning, and how to use Python to conduct a well-designed study to draw your own conclusions from data.

#### Modules
* [**Summary Statistics**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Intro_statistics/Measures%20of%20center%20and%20spread.ipynb)
Summary statistics gives you the tools you need to boil down massive datasets to reveal the highlights. In this repo, I explored summary statistics including   mean, median, and standard deviation, and showed how to accurately interpret them.
* [**Random Numbers and Probability**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Intro_probability/Random_numbers_and_probability.ipynb)
In this repo, I cocered how to generate random samples and measure chance using probability. I also worked with real-world sales data to calculate the probability of a salesperson being successful. Finally, I used the binomial distribution to model events with binary outcomes.
* [**More distributions and the central limit theorem**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Intro_probability/Normal_distribution.ipynb)
I created histograms to plot normal distributions and displayed examples of the central limit theorem, before working on statistical functions by adding the Poisson, exponential, and t-distributions to my repertoire.
* [**Correlation and Experimental Design**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Correlation_and_exp_design/Correlation_and_Experimental_Design.ipynb)
I showed how to quantify the strength of a linear relationship between two variables, and explore how confounding variables can affect the relationship between two other variables. I also showed how a study’s design can influence its results, change how the data should be analyzed, and potentially affect the reliability of conclusions.

## Supervised Learning with Scikit-Learn
Machine learning is the field that teaches machines and computers to learn from existing data to make predictions on new data: Will a tumor be benign or malignant? Which of your customers will take their business elsewhere? Is a particular email spam? In this repo, I covered how to use Python to perform supervised learning, an essential component of machine learning. I also showed how to build predictive models, tune their parameters, and determine how well they will perform with unseen data—all while using real world datasets. I used scikit-learn, one of the most popular and user-friendly machine learning libraries for Python.

#### Modules
* [**Classification**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Supervised_learning/Classification.ipynb)
Classification is the process of predicting the class of given data points. Classes are sometimes called as targets/ labels or categories. For example, spam detection in email service providers can be identified as a classification problem. This is s binary classification since there are only 2 classes as spam and not spam. A classifier utilizes some training data to understand how given input variables relate to the class. In this case, known spam and non-spam emails have to be used as the training data. When the classifier is trained accurately, it can be used to detect an unknown email. Classification belongs to the category of supervised learning where the targets also provided with the input data.
* [**Regression**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Supervised_learning/Introduction_to_regression.ipynb)
Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables. For example, relationship between rash driving and number of road accidents by a driver is best studied through regression.
* **Fine Tuning the Model**
Having trained the model, the next task is to evaluate its performance. Using the metrics available in scikit-learn i showed how to assess my model's performance in a more nuanced manner. I also optimized my classification and regression models using hyperparameter tuning.
* **Preprocessing and Pipelines**
This section introduces pipelines, and how scikit-learn allows for transformers and estimators to be chained together and used as a single unit. Preprocessing techniques was used in enhancing the model performance, and the use of pipelines was also shown. 

## Unsupervised Learning with Scikit-Learn
Say you have a collection of customers with a variety of characteristics such as age, location, and financial history, and you wish to discover patterns and sort them into clusters. Or perhaps you have a set of texts, such as wikipedia pages, and you wish to segment them into categories based on their content. This is the world of unsupervised learning, called as such because you are not guiding, or supervising, the pattern discovery by some prediction task, but instead uncovering hidden structure from unlabeled data. Unsupervised learning encompasses a variety of techniques in machine learning, from clustering to dimension reduction to matrix factorization. In this repo, I covered the fundamentals of unsupervised learning and implementing the essential algorithms using scikit-learn and scipy. I also showed how to cluster, transform, visualize, and extract insights from unlabeled datasets.

#### Modules
* [**Clustering for dataset exploration**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Unsupervised_learning/Clustering.ipynb)
Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other clusters.
* [**Visualization with hierarchical clustering and t-SNE**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Unsupervised_learning/Clustering.ipynb)
In this section, I covered two unsupervised learning techniques for data visualization, hierarchical clustering and t-SNE. Hierarchical clustering merges the data samples into ever-coarser clusters, yielding a tree visualization of the resulting cluster hierarchy. t-SNE maps the data samples into 2d space so that the proximity of the samples to one another can be visualized. 
* **Decorrelating your data and dimension reduction**
Dimension reduction summarizes a dataset using its common occuring patterns. In this section, I covered the most fundamental of dimension reduction techniques, "Principal Component Analysis" ("PCA"). PCA is often used before supervised learning to improve model performance and generalization. It can also be useful for unsupervised learning. For example, employing a variant of PCA will allow you to cluster Wikipedia articles by their content! 
* **Discovering interpretable features**
In this section, I covered a dimension reduction technique called "Non-negative matrix factorization" ("NMF") that expresses samples as combinations of interpretable parts. For example, it expresses documents as combinations of topics, and images in terms of commonly occurring visual patterns. I also used NMF to build recommender systems that can find you similar articles to read, or musical artists that match your listening history! 


# Feature Engineering for Machine Learning in Python
Feature engineering is the process of using data’s domain knowledge to create features that make machine learning algorithms work. It’s the act of extracting important features from raw data and transforming them into formats that are suitable for machine learning. To perform feature engineering, a data scientist combines domain knowledge (knowledge about a specific field) with math and programming skills to transform or come up with new features that will help a machine learning model perform better. Feature engineering is a practical area of machine learning and is one of the most important aspects of it.

#### Modules
[**Creating Features**](https://github.com/Josiah-Jovido/Data_Camp_Data_Science/blob/main/Feature_engineering/Creating_Features.ipynb)
In this repo, I explored what feature engineering is and how to get started with applying it to real-world data. I loaded, explored and visualized a survey response dataset, and in doing so i displayed its underlying data types and why they have an influence on how i engineered my features. Using the pandas package i created new features from both categorical and continuous columns. 
