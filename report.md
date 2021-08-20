# Machine Learning Methods

- What is machine learning and AI?
- Signal and the noise.

===

## 1. Data Analysis

### Data analysis intro

#### Data analysis

#### Unsupervised learning

#### MNIST

#### The curse of dimensionality

#### Dimensionality reduction

---

### 1.1. Principle Component Analysis

#### PCA

what is pca and algorithm

#### Dimensionality reduction with PCA

- explain application to mnist
- various dimensions and recronstructions
- cummulative variance plot
+ try kernel pca
  
#### Dimensionality reduction for visualisation

- why we want to do this
- PCA to 2 dimensions
- compare to tnse

---

### 1.2. Clustering

#### Clustering

what is it and applications (with example)

#### K-Means

What is kmeans and general algorithm

#### Clustering MNIST

- cluster the pca(2) mnist
- voronoi plot
- compare clusters to actual
- analyse silhoutte score
+ finding optimal clusters via elbow

#### Limits of K-Means  

- main downsides to kmeans
- clustering in high dimensional space
- notes on analysing clustering performance

### Conclusion

final analysis of MNIST dataset

===

## 2. Classification

#### Supervised learning

#### Classification

#### Overfitting and Underfitting

#### Training, Testing and Validating

#### Cross-validation

#### Parameters and Optimisation

#### Classification with MNIST

#### No free lunch

### 2.1. Artificial Neural Networks (ANNs)

Train an ANN, plot the training and validation learning curves.  Does the model overfit?  What are your results in the testing dataset?  Interpret and discuss your results.  How do they compare with SVMs?  How do the hyper-parameters (e.g.  learning rate) impact on performance?

#### Neurons

- Neurons structure
- Action potentials ~ activation functions
- Weights, connections etc.

#### The perceptron

- Perceptron history
- Input layer
- Output layer
- Activation function
- Weights
- Bias
- Try and apply it to MNIST (lots of neurons?)

#### Multi-layer perceptron

- Multiple layers of neurons
- Able to learn more complex patterns

#### Training MLPs

- Forward pass
- Backward pass
- Epochs, convergence

#### Forward pass

- Calculation of outputs from input
- Calculates loss

#### Backpropogation

- Gradient descent
- Learning rate
- Minimise loss function
- Convex optimisation problem
- Chain rule

#### Stochastic gradient descent

- What is SGD
- Diagram

#### Cross entropy

- What is cross entropy
- Sparse categorical cross entropy

#### Classifying MNIST

- Feed forward network
- Dense layers
- Epochs, convergence
- Create a simple MLP for MNIST

#### Deeper neural networks

- Create a large overfitting network
- Early stopping, check pointing

#### Vanishing / exploding gradient problem

- What is the vanishing gradient problem

#### Initlialisation

- GloRoot initialisation

#### Saturating activation functions

- Leaky RELU
- SELU
- Tanh...

#### Batching

- Batch normalisation
- Mini-batch
- Implement batching

#### Gradient clipping

- Gradient clipping

#### Optimisation of ANNs

- Momentum
- Adam
- ...

#### Overfitting in ANNs

- Why large networks can overfit
- l1, l2 loss and regularisation
- Implement regularisation
- Dropout, monte carlo dropout
- Create a network with dropout

#### Convolutional neural networks (CNN)

- Problems with neural networks and images
- Convolutions, filters
- Convolutional layers, feature maps
- Simple CNN for MNIST

#### Training better CNNs

- Stride, step
- Pooling, max pooling
- Better CNN

#### Transfer learning

- What is transfer learning
- Use transfer learning with MNIST

### 2.2. Support Vector Machines (SVMs)

Train an SVM (with a chosen Kernel) and perform the same analyses as forANNs. Interpret  and  discuss  your  results. Does the model overfit? How do they compare with ANNs? And why? How does the type of kernel (e.g.linear, RBF, etc.) impact on performance?

#### Support vector machines

- Hyperplane
- Support vectors
- Minimising lagrange multiplier (MIT)

#### Linear SVM

- Implementation of SVM on MNIST
- Plot decision boundaries from lab 2

#### Probabilistic SVM

- What is a probabilistic SVM?
- SVM on MNIST

#### The dual problem

What is the dual problem

#### Kernels

- Different representations
- Kernels
- Mercer's theorem
- Gram matrix

#### The kernel trick

- What is the kernel trick

#### Kernel SVMs

- How kernel SVMs work

#### Polynomial KSVM

- What is the polynomial kernel
- Implementation

#### Radial basis function (RBF) KSVM

- What is the RBF kernel
- Implementation

#### Sigmoid KSVM

- What is the sigmoid kernel
- Implementation

#### Optimising SVMs

- Hinge loss
- Grid search hyperparameters

#### Limitations of SVMs

- Problems with SVMs

### Classification conclusion

#### Comparison of ANNs and SVMs

- Comparison of neural networks and support vector machines.
- Area under ROC curve
- Precision, recall, accuracy...
- TP, TN, FP, FN and rates for each
- Confusion matrix
- Metric
- Validation
- Cross-validation
- Prediction
- Inference
- Interpretability
- Sensitivity vs specificity

## 3. Regression

### Regression intro

#### Regression

- What is regression
- Supervised learning

#### Linear regression

- Linear model
- Common approaches
  - Least squares (frequentist)
  - Bayesian linear regression

#### California housing dataset

- Download data
- Put in pandas data frame
- Analysis of dataset
  - Size, shape, type
  - Features
  - Target feature of dataset

#### Geographical data

- Plot geographical data (with Plotly?)

#### Correlation and distribution

- Look at correlations between features
- PCA?

#### Missing data

- Check for missing data
- Handle missing data

#### Outliers

- Check for outliers
- Handle / justify outliers

#### Feature engineering and selection

- Feature engineering
- Feature selection
- Feature set

#### Categorical features

- Handle categorical features
- One-hot encoding?

#### Notes on sampling

- In group bias
- Ground truth
- Prior belief
- Sampling bias
- Selection bias
- Bias variance trade-off

### 3.1 Bayesian Linear Regression

#### Bayes Theorem

- Bayes theorem
- Bayes rule
- Bayesian statistics

#### MCMC

- Markov chain
- Markov property
- MCMC
- Metropolis Hastings

#### Bayesian linear regression

- What is bayesian linear regression

#### Prior beliefs

- Prior for Bayesian
- Log-normal distribution

#### Prepare data for model

- Test train val split
- Scaling

#### Pipeline

- Create pipeline

#### Performance measure

- Define the performance measure of the model.
- MSE / RMSE

#### Training the model

- Fit linear regressor

#### Evaluation of model

- Evaluate model performance on val / test set

#### Bayesian linear regression optimisation

- Hyper parameter tuning
- Feature engineering
- Feature selection
- Normalisation
- Distance to LA / SF
- Log of dataset
- Handle cut-offs
- Ridge / lasso regression

### Regression conclusion

- Concluding points on things learnt from dataset.
- Concluding points of Bayesian regression performance.

## 4. Ensembles

### Ensemble intro

#### Wisdom of the crowds

- Wisdom of the crowds

#### Ensemble methods

- What is ensemble learning?
- Voting classifiers
- Weak learners etc.

#### Bagging

- What is bagging
- Bagging and pasting
- Out of bag evaluation

#### Random bagging 

- What is random bagging
- Random patches and subspaces

### 4.1 Random Forest

#### Decisions trees

- Create a decision tree

#### Random forest

- Bag a decision tree
- Random forrest regressor

#### Extra trees

- Extremely randomised trees

#### Random forrest optimisation

- Hyperparameter optimisation
- Grid search

#### Tasks

Analyse the effect of the hyperparameters of the random forest, such as the number of estimators (or base models, i.e. the number of decision trees that are combined into the random forest). Look at the constructor of the RandomForestRegressor class to see what hyperparameters you can set. In your analysis, include the following plots and discussions but you may wish to add further analysis of your own:

1. Plot the relationship between a hyperparameter and the performance of the model.
2. Optimise the hyperparameter on a validation set.
3. Plot the trade-off between time taken for training and prediction performance.
4. What do you think is a good choice for the number of estimators on this dataset?
5. What is the effect of setting the maximum tree depth or maximum number of features?
6. Is the random forest interpretable? Are the decision trees that makeup the forest interpretable?

### 4.2 Stacking

#### Stacking

- What is stacking

#### Stacked regressors

- Stack Bayesian regression with decision trees

#### Stacking tasks

Bayesian linear regression and decision trees are two very different approaches to regression.  Ensemble methods can exploit such diversity between different methods to improve performance.  So now you will try combining the random forest  and  Bayesian  linear  regression  using stacking.

Scikit-learn  includes the  StackingRegressor  class  to  help  you  with  this.   In  the  report,  explain the stacking approach and describe your results,  making sure to cover the following points:

1. When does stacking improve performance over the individual models (e.g. try stacking with a random forest with 'maxdepth=10' and 'nestimators=10')?
2. What happens if we just take the mean prediction from our base models instead?
3. Use a DecisionTreeRegressor as the final estimator and visualise the tree to understand what stacking is doing.

#### Boosting

- What is boosting

#### Gradient boosting

- What is gradient boosting?

#### XGBoost

- Implement XGBoost regression for housing data

#### Stacking gradient

- Try stacking XGBoost regressors

### Ensemble Conclusion

#### Comparison of random forest and stacking

- Comparison of random forest and stacking methods
  - Accuracy
  - Computational complexity
  - Data needed
  - Generalisation / ease of use
