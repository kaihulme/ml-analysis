# Machine Learning Methods

#### Machine learning

- What is machine learning and AI?
- Signal and the noise.

## 1. Data Analysis

#### Data analysis
- What is data?
- What is data analysis?

#### Unsupervised learning
- What is unsupervised learning?

#### MNIST
- Initial analysis MNIST dataset
  - Size, shape, type
  - Features
  - Missing data, outliers etc.
  - Target feature of dataset (class)

#### The curse of dimensionality
- Problems with high dimension data.
- Relate to MNIST.
- Decribe the curse of dimensionality.

#### Dimensionality reduction
- What is dimensionality reduction
- The aim of dimensionality reduction.
- Main approaches:
  - Projection
  - Manifold
- Common methods:
  - PCA
  - t-SNE
  - Autoencoding
  - ??
  
### 1.1 Principle Component Analysis

#### PCA
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
- PCA to 2 dimensions
  - Scatter plot
  - Interpret results

#### Other dimensionality reduction methods for visualisation
  - t-SNE
    - PCA then t-SNE
    - Autoencoding?

### 1.2 Clustering

#### Clustering
- What is clustering?
- Applications
- Different approaches
  - KMeans
  - Gaussian mixtures

#### K-Means
- What is K-Means?
  - Idea
  - Algorithm

#### Clustering and visualising MNIST
- 10 clusters, 2 dimensions
  - plot cluster centroids on plot
  - veronoi graph
  - accuracy

#### Finding the optimal number of clusters
  - inertia / elbow plot
  - silhouette score
  - silhouette diagram
  
#### Clustering in high-dimensional space
  - various dimensions
  - various kernels

#### Limits of K-Means  
- limits of kmeans

#### Other clustering methods
  - kmedian
  - analyse gaussian mixures
  - bayesian gaussian mixtures
  - dbscan?

### Conclusion
- Final analysis of MNIST dataset

## 2. Classification

#### Supervised learning
- What is supervised learning?
- Training
- Fitting
- Labels
- Error / cost / objective functions

#### Classification
- What is classification?
- Different approaches
  - Logistic regression - log loss
  - ANNs, SVMs
  - ...

#### No free lunch
- Model selection.
- Parametric / nonparametric models.
- Discriminitive / generative modelling.

#### Overfitting
- Overfitting
- Underfitting
- Generalisation

#### Optimisation
- Hyperparameters

#### MNIST
- The aim of classification with MNIST
- Recap analysis
  - Size, shape, type
  - Features
  - Missing data, outliers etc.
  - Target feature of dataset (class)

#### Preparing the data
- Get the data
  - train, val test sets
  - cross validation
  - reshape / retype

### 2.1 Artificial Neural Networks (ANNs)

Train an ANN, plot the training and validation learning curves.  Does themodel overfit?  What are your results in the testing dataset?  Interpret anddiscuss your results.  How do they compare with SVMs?  How do the hyper-pameters (e.g.  learning rate) impact on performance?

#### Neurons
- Neurons structure
- Action potentials ~ activation functions
- Weights, connections etc.

#### The perceptron
- Perceptron history
- Input layer
- Output laer
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
- Sparse categorical crossentropy

#### Classifying MNIST
- Feed forward network
- Dense layers
- Epochs, convergence
- Create a simple MLP for MNIST

#### Deeper neural networks
- Create a large overfitting network
- Early stopping, checkpointing

#### Vanishing / exploding gradient problem
- What is the vanishing gradient problem

#### Initlialisation
- Gloroot initialisation

#### Saturating activation functions
- Leaky relu
- Selu
- Tanh...

#### Batching
- Batch normalisation
- Mini-batch
- Implement batching

#### Gradient clipping
- Gradient clipping

#### Optimisation
- Momentum
- Adam
- ...

#### Overfitting
- Why large networks can overfit
- l1, l2 loss and regularisation
- Implement regularisation
- Dropout, montecarlo dropout
- Create a network with dropout

#### Convolutional neural networks (CNN)
- Problems with neural networks and images
- Convolutions, filters
- Convolutional layers, feature maps
- Simple CNN for MNIST

#### Training better CNNs
- Stride, step
- Pooling, maxpooling
- Better CNN

#### Transfer learning
- What is transfer learning
- Use transfer learning with MNIST

### 2.2 Support Vector Machines (SVMs)

Train an SVM (with a chosen Kernel) and perform the same analyses as forANNs.   Interpret  and  discuss  your  results.   Does  the  model  overfit?   Howdo they compare with ANNs?  And why?  How does the type of kernel (e.g.linear, RBF, etc.) impact on performance?

#### Support vector machines
- Hyperplane
- Support vectors
- Minimising legrange multiplier (MIT)

#### Linear SVM
- Implementation of SVM on MNIST

#### Probabilistic SVM
- What is a probabilistic SVM?
- PSVM on MNIST

#### The dual problem
What is the dual problem

#### Kernels
- Different representations
- Kernels
- Mercer's thoerum
- Gram matrix

#### The kernel trick
- What is the kernel tricl

#### Kernel SVMs
- How kernel SVMs work

#### Polynomial KSVM
- What is the polynomial kernel
- Implementation

#### RBF KSVM
- What is the RBF kernel
- Implementation

#### Sigmoid KSVM
- Whatis the sigmoid kernel
- Implementation

#### Optimising SVMs
- Hinge loss
- Grid search hyperparameters

#### Limitations of SVMs

### Comparison of Methods
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

## 3. Regression

#### Regression
- What is regression
- Supervised learning

#### Linear regression
- Linear model
- Common approaches
  - Least squares
  - Bayesian linear regression

#### California housing dataset
- Download data
- Put in pandas dataframe
- Analysis of dataset
  - Size, shape, type
  - Features
  - Target feature of dataset

#### Geographical data
- Plot geographical data (with plotly?)

#### Correlations
- Look at correlations between features
- PCA?

#### Missing data
- Check for missing data
- Handle missing data

#### Outliers
- Check for outliers
- Handle / justify outliers

#### Feature engineering
- Feature engineering
- Feature selection
- Feature set

#### Categorical features
- Handle categorical features
- One-hot encoding?

#### Pipeline
- Create pipeline

#### Notes on sampling
- in group bias
- ground truth
- prior belief
- sampling bias 
- selection bias

### Bayesian Linear Regression

#### Bayes Theorum
- Bayes theorum
- Bayes rule
- Bayesian statistics

#### Prior beliefs
- Prior for Bayesian

#### MCMC
- Markov chain
- Markov property
- MCMC 
- Metropolis Hastings

#### Bayesian linear regression
- Fit linear regressor

#### Optimisation
- Hyper parameter tuning
- Feature engineering
- Feature selection
- Normalisation

### Conclusion
- Concluding points on things learnt from dataset.
- Concluding points of Bayesian regression performance.

## 4. Ensemble Learning

# Wisdom of the crowds
- Wisdom of the crowds

#### Ensemble methods
- What is ensemble learning?
- vVting classifiers
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

#### Optimisation
- Hyperparameter optimisation
- Gridsearch

#### Tasks
Analyse the effect of the hyperparameters of the random forest, such as the number of estimators (orbase models, i.e. the number of decision treesthat are combined into the random forest). Look at the constructor of theRandomForestRegressor class to see what hyperparameters you can set. In your analysis, include the following plots and discussions but you may wish to add further analysis of your own:

1. Plot the relationship between a hyperparameter and the performanceof the model.
2. Optimise the hyperparameter on a validation set.
3. Plot the trade-off between time taken for training and prediction per-formance.
4. What do you think is a good choice for the number of estimators onthis dataset?
5. What is the effect of setting the maximum tree depth or maximum number of features?
6. Is the random forest interpretable? Are the decision trees that makeup the forest interpretable?

### 4.2 Stacking

#### Stacking
- What is stacking

#### Stacked regressors
- Stack Bayesian regression with decision trees

#### Tasks
Bayesian linear regression and decision trees are two very different approachesto regression.  Ensemble methods can exploit such diversity between differentmethods to improve performance.  So now you will try combining the randomforest  and  Bayesian  linear  regression  usingstacking.   Scikit-learn  includesthe  StackingRegressor  class  to  help  you  with  this.   In  the  report,  explainthe stacking approach and describe your results,  making sure to cover thefollowing points:

1. When does stacking improve performance over the individual models (e.g. try stacking with a random forest with maxdepth=10 and nestimators=10)?
2. What happens if we just take the mean prediction from our base models instead?
3. Use a DecisionTreeRegressor as the finalestimator and visualise the tree to understand what stacking is doing.

#### Boosting
- What is boosting

#### Gradient boosting
- What is gradient boosting?

#### XGBoost
- Implment XGBoost regression for housing data

#### Stacking gradient
- Try stacking XGBoost regressors

### Comparison of Methods
- Comparison of random forest and stacking methods
  - Accuracy
  - Computational complexity
  - Data needed
  - Generalisation / ease of use