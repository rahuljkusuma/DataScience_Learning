import sklearn
import pandas as pd


housing = pd.read_csv("data.csv")

housing.head()
housing.shape
housing.info()
housing.CHAS
housing.CHAS.value_counts()
housing.describe()

# %matplotlib inline
# get_ipython().run_line_magic('matplotlib', 'inline')

""" For plotting histogram """
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(25,25))
plt.show()




""" ==>Train-Test Splitting<== """

""" One way of splitting training and testing data """
import numpy as np
# It is just for learning purpose
# def split_train_test(data, test_ratio):
#     np.random.seed(42) # Used to shuffle the data only once.
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# test_ratio = 20/100
# train_set, test_set = split_train_test(housing, test_ratio)

"""" Second Option Is from Scikit-Learn """
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in train set: {len(test_set)}")

# This split with sklearn will work just fine.
# 
# But..!
# 
# But...!
# 
# But....!
# 
# There is a problem. For example just looke at "housing.CHAS.value_counts()" this is giving counts of different values in CHAS. CHAS is having 471 zeros(0s) and 35 ones(1s).
# What if all the ones(1s) go inside test_set? Then our machine will not be trained for ones(1s).
# So to avoid this we have to follow below method.

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing.CHAS):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set.CHAS.value_counts() # 95/7
strat_train_set.CHAS.value_counts() # 376/28



""" Now assign housing as our main training Dataset """
housing = strat_train_set.copy()



""" ==>Looking for Correlations<== """

corr_matrix = housing.corr()
corr_matrix.MEDV.sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["MEDV", "RM","ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(24,16))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)
plt.show()



# ## Trying Out Attribute Combinations

# housing['TAXRM'] = housing.TAX/housing.RM
# housing.head()

corr_matrix = housing.corr()
corr_matrix.MEDV.sort_values(ascending=False)

# housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()



""" ==>Missing Attributes<== """

""" To take care of missing attributes, you have three options: """
# 1. Get rid of the missing data points:
# >Removing entire row. This we can do wherever the missing data points are 2 or 3 or less than 0.5% of the data.
# 2. Get rid of the whole attribute
# >Removing entire column. If correlation coefficient is very less or near to zero, then we can remove entire column.
# 1. Set the value to some value(0, mean or median)
# 
""" Please note that have to do these operations with training set and NOT with entire dataframe. """

# Option#1
a = housing.dropna(subset=["RM"])
a.shape
# Note that the original hosuing dataframe will remain unchanged

# Option#2
b = housing.drop("RM", axis=1)
b.shape
# Note that the original hosuing dataframe will remain unchanged

# Option#3-1
median = housing.RM.median()
print(median)
housing.RM.fillna(median)
# Note that the original hosuing dataframe will remain unchanged
# housing.describe()

# Option#3-2 another way of filling empty cells or missing data with median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

imputer.statistics_.shape

X = imputer.transform(housing)

housing_tr = pd.DataFrame(X, columns=housing.columns)
# housing_tr.describe()



""" ==>Scikit-learn Design<== """ 

""" Primarily, there are three types of objects: """ 
# 1. Estimators: It estimates some parameters based on a dataset. Eg. imputer.
# >It has a fit method and transform method.<br>
# >Fit method - Fits the dataset and calculates internal parameters.
# 
# 2. Transformers:
# >Transform method takes input and returns output based on the learnings from fit().<br>
# >It also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors:
# >LinearRegression model is an example of predictor. fit() and predict() are two common functions.<br>
# >It also gives score() function which will evaluate the predictions.



""" ==>Feature Scaling<== """

""" Primarily, two types of feature scaling methods: """
# 1. Min-max scaling (Normalization):
# >Formula is ((value-min)/(max-min))<br>
# >For this Sklearn provides a class MinMaxScaler for this.
# 2. Standardization(Z-score):
# >Formula is ((value-mean)/std)<br>
# >For this Sklearn provides a class Standard Scaler for this.


"""Creating a Pipeline"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    # ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)

housing_num_tr.shape



""" Selecting A Desired Model For Dragon Real Estates """

from sklearn.linear_model import LinearRegression

fmodel = LinearRegression() #Failed Model
fmodel.fit(housing_num_tr, housing_labels) #Failed Model


from sklearn.tree import DecisionTreeRegressor

f1model = DecisionTreeRegressor() #Failed Model
f1model.fit(housing_num_tr, housing_labels) #Failed Model

model = DecisionTreeRegressor()
model.fit(housing_num_tr, housing_labels)


from sklearn.ensemble import RandomForestRegressor

f2model = RandomForestRegressor()
f2model.fit(housing_num_tr, housing_labels)



some_data = housing.iloc[:5]

some_labels = housing_labels[:5]

prepared_data = my_pipeline.transform(some_data)

fmodel.predict(prepared_data)

f1model.predict(prepared_data)

f2model.predict(prepared_data)

model.predict(prepared_data)


list(some_labels)


# ## Evaluating The Model

from sklearn.metrics import mean_squared_error

fhousing_predictions = fmodel.predict(housing_num_tr)
fmse = mean_squared_error(housing_labels, fhousing_predictions)
frmse = np.sqrt(fmse)
print(f"Linear Regression Model\nmse:- {fmse} and rmse:-{frmse}") 
#Not good model because of underfitting


f1housing_predictions = f1model.predict(housing_num_tr)
f1mse = mean_squared_error(housing_labels, f1housing_predictions)
f1rmse = np.sqrt(f1mse)
print(f"Decision Tree Regressor Model\nmse:- {f1mse} and rmse:-{f1rmse}") 
#Not good model because of overfitting


f2housing_predictions = f2model.predict(housing_num_tr)
f2mse = mean_squared_error(housing_labels, f2housing_predictions)
f2rmse = np.sqrt(f2mse)
print(f"Random Forest Regressor model\nmse:- {f2mse} and rmse:-{f2rmse}")


housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
mse #Not good model because of overfitting


""" ==> Using Better Evaluation Technique - Cross Validaion<== """

from sklearn.model_selection import cross_val_score

# Cross validation on DecisionTreeRegressor Model
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

# Cross validation on LinearRegression Model
fscores = cross_val_score(fmodel, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
frmse_scores = np.sqrt(-fscores)
frmse_scores

# Cross validation on RandomForestRegressor Model
f2scores = cross_val_score(f2model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
f2rmse_scores = np.sqrt(-f2scores)
f2rmse_scores


def print_score(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())

# Decision Tree Model Output
print_score(rmse_scores)

# Linear Regression Model Output
print_score(frmse_scores)

# Random Forest Regressor Model Output
print_score(f2rmse_scores)
