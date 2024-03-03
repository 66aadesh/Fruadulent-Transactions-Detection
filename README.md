# Fuadulent-Transactions-Detection
Detect fraudulent transactions using ML models (Logistic Regression, KNN &amp; Random Forest). Clean data, analyze, &amp; apply models for accurate predictions. Observations highlight model biases &amp; performances for better tuning.

### Content

1. [Import necessary Libraries](#1-import-necessary-libraries)
2. [Import the Data](#2-import-the-data)
3. [EDA & Cleaning on Data](#3-eda--cleaning-on-data)
4. [Plotting graphs to visualize the relation between the parameters and Fraudulent transactions](#4-plotting-graphs-to-visualize-the-relation-between-the-parameters-and-fraudulent-transactions)
5. [Deleting Outliers](#5-deleting-outliers)
6. [Testing multicollinearity](#6-testing-multicollinearity)
7. [Applying ML models to predict the required values](#7-applying-ml-models-to-predict-the-required-values)
8. [Observations](#8-observations)

### 1. Import necessary Libraries
```python
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2. Import the Data

### 3. EDA & Cleaning on Data
See the code and descriptions in the file.

### 4. Plotting graphs to visualize the relation between the parameters and Fraudulent transactions
See the code and descriptions in the file.

### 5. Deleting Outliers
See the code and descriptions in the file.

### 6. Testing multicollinearity
See the code and descriptions in the file.

### 7. Applying ML models to predict the required values
See the code and descriptions in the file.

### 8. Observations
See the code and descriptions in the file.

**ML Modelling Observations:**

- The accuracy scores for all three models are quite high, indicating that the models perform well in terms of overall classification accuracy.
- The confusion matrix provides a more detailed view of the model performance:
  - **Logistic Regression**: It correctly classifies a large number of instances with label 0, but it fails to predict any instances with label 1. This suggests that the model might be biased towards the majority class.
  - **K Nearest Neighbors**: It has a similar number of false negatives (instances wrongly classified as 0) as Logistic Regression, but it correctly classifies more instances with label 1.
  - **Random Forest**: It performs better than Logistic Regression in terms of capturing instances with label 1, but it misclassifies more instances with label 0 compared to Logistic Regression.
- In summary, while all three models have high accuracy, K Nearest Neighbors seems to strike a better balance in correctly classifying instances of both classes. Random Forest, although having a slightly lower accuracy than the other two models, shows better performance in capturing instances of the minority class. Further analysis and fine-tuning may be necessary to address the biases observed in Logistic Regression and to optimize the performance of all three models.
