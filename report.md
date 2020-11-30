# Machine Learning Methods

## 1. Data Analysis

### 1.1 Analysing data
- What is data?
- What is data analysis?
- What is unsupervised learning?
- Initial analysis MNIST dataset
  - Size, shape, type
  - Features
  - Missing data, outliers etc.
  - Target feature of dataset (class)

### The Curse Of Dimensionality
- Problems with high dimension data.
- Relate to MNIST.
- Decribe the curse of dimensionality.

### Dimensionality Reduction
- What is dimensionality reduction
- The aim of dimensionality reduction.
- Main approaches:
  - Projection
  - Manifold
  
### Principle Component Analysis

#### PCA:
- What is PCA?
  - Idea
  - Algorithm

#### Dimensionality reduction:
- Application to MNIST
  - Projecting down to various dimensions
  - Analysing variance ratio
    - Few different component amounts
    - Cummulative variance plot
  - Reconstructions of compressed images
    - Various compression amounts

#### Other PCA methods:
- Different types of PCA
  - Normal / linear
  - Probablistic
  - Random
  - Incremental
  - Kernel
    - Various kernels
  
#### Visualisation with PCA
- Data visualisation
  - PCA to 2 dimensions
    - Scatter plot
    - Interpret results
  - t-SNE
    - PCA then t-SNE
    - Autoencoding?

### 1.2 Clustering

- What is clustering
- Applications
- Different approaches
  - KMeans
  - Gaussian mixtures

### K-Means


- 10 clusters, 2 dimensions
  - veronoi graph
  - accuracy

- finding the optimal clusters (10)
  - inertia / elbow plot
  - silhouette score
  - silhouette diagram
  
- higher-dimension clustering
  - various dimensions
  - various kernels
  
- limits of kmeans

  - analyse gaussian mixures
  - bayesian gaussian mixtures
  - dbscan?

### Conclusion

- Final analysis of MNIST dataset

## 2. Classification

- supervised learning
- get the data
  - train, val test sets
  - cross validation

### 2.1 Artificial Neural Networks (ANNs)

Train an ANN, plot the training and validation learning curves.  Does themodel overfit?  What are your results in the testing dataset?  Interpret anddiscuss your results.  How do they compare with SVMs?  How do the hyper-pameters (e.g.  learning rate) impact on performance?

### 2.2 Support Vector Machines (SVMs)

Train an SVM (with a chosen Kernel) and perform the same analyses as forANNs.   Interpret  and  discuss  your  results.   Does  the  model  overfit?   Howdo they compare with ANNs?  And why?  How does the type of kernel (e.g.linear, RBF, etc.) impact on performance?

### Comparison of Methods

- Comparison of neural networks and support vector machines.

## 3. Regression

- What is regression
- Analysis of dataset
  - Size, shape, type
  - Features
  - Missing data, outliers etc.
  - Target feature of dataset

### Bayesian Linear Regression

In this task you are required to use PyMC3 to perform Bayesian linear re-gression on the California housing dataset which is easily available via thesklearn.datasets.fetchcaliforniahousing function.  The goal with this datasetis to predict the median house value in a ‘block’ in California.  A block isa small geographical area with a population of between 600 and 3000 peo-ple.   Each  datapoint  in  this  dataset  corresponds  to  a  block.   Consult  thescikit-learn documentation for details of the predictor variables. As  always  with  Bayesian  analysis  it  is  up  to  you  to  choose  your  priordistributions.  Be sure to justify your choice of priors in your report.  Whatdo  the  results  produced  by  PyMC3  tell  you  about  what  influences  housevalue in California?  Is it necessary and/or useful to transform the data insome way before running MCMC?

### Conclusion

- Concluding points on things learnt from dataset.
- Concluding points of Bayesian regression performance.

## 4. Ensemble Learning

- What is ensemble learning?
- Wisdom of the crowds, weak learners etc.

### 4.1 Random Forest

This part builds on the related lab (week 7). First, run a random forestregressor for the California housing dataset, and contrast this with your previous Bayesian linear regression method. For this you can use the Ran-domForestRegressor class from Scikit-learn.

Analyse the effect of the hyperparameters of the random forest, such as the number of estimators (orbase models, i.e. the number of decision treesthat are combined into the random forest). Look at the constructor of theRandomForestRegressor class to see what hyperparameters you can set. In your analysis, include the following plots and discussions but you may wish to add further analysis of your own:

1. Plot the relationship between a hyperparameter and the performanceof the model.
2. Optimise the hyperparameter on a validation set.
3. Plot the trade-off between time taken for training and prediction per-formance.
4. What do you think is a good choice for the number of estimators onthis dataset?
5. What is the effect of setting the maximum tree depth or maximum number of features?
6. Is the random forest interpretable? Are the decision trees that makeup the forest interpretable?

### 4.2 Stacking

Bayesian linear regression and decision trees are two very different approachesto regression.  Ensemble methods can exploit such diversity between differentmethods to improve performance.  So now you will try combining the randomforest  and  Bayesian  linear  regression  usingstacking.   Scikit-learn  includesthe  StackingRegressor  class  to  help  you  with  this.   In  the  report,  explainthe stacking approach and describe your results,  making sure to cover thefollowing points:

1. When does stacking improve performance over the individual models (e.g. try stacking with a random forest with maxdepth=10 and nestimators=10)?
2. What happens if we just take the mean prediction from our base models instead?
3. Use a DecisionTreeRegressor as the finalestimator and visualise the tree to understand what stacking is doing.