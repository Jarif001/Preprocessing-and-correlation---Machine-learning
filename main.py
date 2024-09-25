import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Employee_Attrition.csv')

rows = dataset.shape[0]
columns = dataset.shape[1]
print('Number of attributes (columns) ' + str(columns))
print('Number of records (rows) ' + str(rows))

# Show the statistics of the dataset ( column wise mean, standard deviation...)
stats_all_data = dataset.describe()
print()
print('Statistics of dataset')
print(stats_all_data)

# Count the number of missing values in the dataset
missing_value_count = dataset.isnull().sum()
print('Missing value count')
print(missing_value_count)
print()

# count the number of duplicate values
duplicate_value_count = dataset.duplicated().sum()
print('Duplicate value count')
print(duplicate_value_count)
print()

# replace missing values with mean
# dataset.fillna(dataset.mean(), inplace=True)
# it won't work coz there are string data types. Do after converting to numeric

# one copy of the duplicates
dataset.drop_duplicates(inplace=True)
# print(dataset.shape)

# drop the row of the target col which is null
dataset.dropna(subset=['Attrition'], inplace=True)
# print(dataset.shape)

# split input and output (features and target)
target = dataset['Attrition']
features = dataset.drop(columns=['Attrition'])
labels = dataset.columns
feature_labels_initial = features.columns

# print(target.tolist().count('No'))

# conversion of target to numeric values by label encoding (int values)
encoder = LabelEncoder()
target = encoder.fit_transform(target)
# print(target)

# conversion of features by one-hot encoding
features = pd.get_dummies(features)
# print(features.head(10))

# replace missing values of features with mean
features.fillna(features.mean(), inplace=True)


# scaling the features - standard (choice 0), min-max (choice 1)
def scale_features(the_features, choice):
    feature_scaled = the_features.copy()
    # have to avoid scaling one-hot (dummy) columns
    one_hot_dummy_columns = feature_scaled.columns.difference(feature_labels_initial)
    scaling_columns = feature_scaled.columns.difference(one_hot_dummy_columns)
    if choice == 0:
        scaler = StandardScaler()
        feature_scaled[scaling_columns] = scaler.fit_transform(the_features[scaling_columns])
    else:
        scaler = MinMaxScaler()
        feature_scaled[scaling_columns] = scaler.fit_transform(the_features[scaling_columns])
    return feature_scaled


# scaled_features_std = scale_features(features, 0)
# scaled_features_minmax = scale_features(features, 1)
choice_ = 0
scaled_features = scale_features(features, choice_)

# convert from array to dataset again with the main columns
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
# print(scaled_features.isnull().sum())

# correlation with the target column
# construct target dataframe with the values of target
target_df = pd.DataFrame(target, columns=['Attrition'])
# print(target_df)
target_series = target_df['Attrition']
# print(target_series)

correlations = scaled_features.corrwith(target_series)
# the above line caused the runtime error. there are some constant values in some
# columns. So their deviation is zero. Hence, dividing causes error. That corr is NaN
# print(correlations)

# top 20 highest correlations
correlations = correlations.abs()
correlations.sort_values(ascending=False, inplace=True)
corr_top20 = correlations.head(20)
# print(corr_top20)

# for plotting graph
# standard scaling part
class0 = scaled_features.loc[target_df['Attrition'] == 0]
class1 = scaled_features.loc[target_df['Attrition'] == 1]
# print(class1)
# print(class0[0])

top20col = corr_top20.index
# print(top20col)

for col in top20col:
    plt.plot(class0[col], np.zeros_like(class0[col]), 'o', label='No')
    plt.plot(class1[col], np.zeros_like(class1[col]), 'o', label='Yes')
    plt.legend()
    plt.xlabel(col)
    plt.title('1D scatter plot of ' + col)
    plt.show()

# bonus (pipeline)
scaled_features = scaled_features[top20col]  # take the top 20 cols
# print(scaled_features)

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target_series, test_size=0.2, random_state=42)

# Step 2: Initialize the Logistic Regression classifier
clf = LogisticRegression()

# Step 3: Train the classifier on the training data
clf.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_prediction = clf.predict(X_test)

# Step 5: Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_prediction)
print(f"Accuracy of Logistic Regression classifier: {accuracy:.2f}")
