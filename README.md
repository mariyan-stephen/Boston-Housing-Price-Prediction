# Boston Housing Price Prediction

This project aims to predict housing prices in Boston using the Boston Housing dataset from the UCI Machine Learning Repository. The dataset is also included in the scikit-learn library, making it easy to load and use for experimentation.

## Dataset

The Boston Housing dataset consists of 506 samples with 13 features and a target column (PRICE). The features are:

1. CRIM: per capita crime rate by town
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS: proportion of non-retail business acres per town
4. CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. NOX: nitric oxides concentration (parts per 10 million)
6. RM: average number of rooms per dwelling
7. AGE: proportion of owner-occupied units built prior to 1940
8. DIS: weighted distances to five Boston employment centers
9. RAD: index of accessibility to radial highways
10. TAX: full-value property-tax rate per $10,000
11. PTRATIO: pupil-teacher ratio by town
12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
13. LSTAT: % lower status of the population

The target column, PRICE, represents the median value of owner-occupied homes in $1000s.

## Requirements

- Python 3.7+
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Usage

1. Load the dataset using the scikit-learn library.
2. Perform exploratory data analysis (EDA) using correlation heatmaps.
3. Select features and split the data into training and testing sets.
4. Train a linear regression model on the training set.
5. Make predictions using the test set and evaluate the model's performance.
6. Visualize the predicted prices against the actual prices.

## Results

The performance of the linear regression model can be assessed using metrics such as mean squared error, root mean squared error, and R-squared score. The predicted prices can be visualized against the actual prices using a scatter plot.

