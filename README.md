# Module 5 - Predicting Holiday Flight Delays

## Business Understanding
With the holiday season approaching, it's important to make the best plans for air travel. We will build a model to predict whether or not a flight will be delayed during Year-End Travel Season, spanning December 21-January 2. For the purpose of this study, we will consider the period from December 21-December 31, as January 2019 data is not available. The top three US airlines (by revenue passenger-kilometers) were chosen to make the amount of data manageable. This study includes Delta, United and American Airlines. 

## Zillow Economic Research Dataset
The Kaggle dataset contains information on flight delays and cancellation for 2018 from the Bureau of Transportation Statistics. There are over 7 millions records for on-time flight performance for US domestic flights. 

## Approach

1. Data imported and cleaned to remove or replace missing values and outliers
2. Data for Top 3 Airlines (Delta Airlines, United Airlines, American Airlines) extracted
3. Data for Holiday Travel Season (December 21-31) extracted 
4. Four Classification Models built using Pipeline with GridSearchCV
    - K Neaest Neighbors
    - Random Forest
    - XGBoost
    - Support Vector Machines
5. Best model parameters refitted during GridSearchCV
6. Best performing model selected using accuracy score
7. Top 5 most important features determined using feature importance

## Files

- README.md: Instruction guide
- Airline Delay Predictions Pipeline.ipynb : a jupyter notebook file
- 2018.csv : Contains flight statistics for US domestic airlines for 2018
- Worst-Christmas-Travel-Dates.png: an image file that projects worst 2019 holiday travel dates
- Holiday Travel Forecast.png: an image file predicting 2019 Holiday Travel forecast
- worst days.jpg: an image file that contains travel stats for Holiday Season 2018
- XGB Feature Importance.png: an image file that contains XGBoost Feature Importance
- Predicting Holiday Flight Delays.pdf: presentation slides that hightlight the results from this project
- cf_matrix.py: file containing function to make plot an sklearn Confusion Matrix cm using a Seaborn heatmap visualization

## Requirements

In order to run the code in the jupyter notebook, you must have Python installed on you computer. Use Anaconda, as it had many useful libraries pre-installed. 

## Installation

Import the following libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn')

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc

from cf_matrix import make_confusion_matrix

import warnings
warnings.filterwarnings("ignore")
```

## Findings

According to our models, we chose XGBoost for predictions with an accuracy score of 93.7%. 

Best Model Parameters:

* **Colsample Bytree**: 0.7
* **Max Depth**: 10
* **Number of Estimators**: 40
### Confusion Matrix

<img src= 'XGB Confusion Matrix.png' height=50% width=50%>

The top 5 most important features with importances scores are below:

* **DISTANCE**: 0.23953465
* **DEST**: 0.19549185
* **ORIGIN**: 0.18115716
* **DEP_DELAY**: 0.09753817
* **TIME_OF_DAY_ARR**: 0.07925626
