import kagglehub
import matplotlib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Set overall plot style
sns.set(style='whitegrid')

# Download latest version
path = r"C:\Users\OlanipeA\Documents\EPL_stat_data.csv"
dataset = pd.read_csv(path)

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

#print(dataset.head())

#Check data types
#print(dataset.dtypes)

# Convert the 'date' column to datetime
dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")

# Remove comma from attendance values
dataset["attendance"] = dataset["attendance"].str.replace(",", "").astype(float)

# Check data types after cleaning
#dataset.info()

# Histograms for selected numeric columns
numeric_cols = ["attendance", "Goals Home", "Away Goals", 'home_possessions',
                'away_possessions','home_shots', 'away_shots', "home_chances", "away_chances"]
# for col in numeric_cols:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(dataset[col].dropna(), kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.tight_layout()
    #plt.show()


# Create a correlation heatmap for numeric columns if there are 4 or more
# numeric_dt = dataset.select_dtypes(include=[np.number])
# if numeric_dt.shape[1] >= 4: # runs if there are at least 4 numeric columns!
#     plt.figure(figsize=(12, 10))
#     corr = numeric_dt.corr()
#     sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
#     plt.title('Correlation Heatmap of Numeric Features')
#     plt.tight_layout()
#     plt.show()


# Pair plot for a subset of numeric features
# subset_cols = ['Goals Home', 'Away Goals', 'home_shots', 'away_shots']
# sns.pairplot(dataset[subset_cols].dropna())
# plt.suptitle('Pair Plot of Selected Features', y=1.02)
# plt.show()

# Testing the above code_block
# sub_cols = ['home_possessions', 'away_possessions', "home_chances", "away_chances"]
# sns.pairplot(dataset[sub_cols].dropna())
# plt.suptitle('Plot of Selected Features', y=1.02)
# plt.show()

# Count plot for the 'class' column to see distribution (if class categories are interesting)
# plt.figure(figsize=(8, 4))
# sns.countplot(data=dataset, x='class')
# plt.title('Count Plot for Class')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# To know the number of classes in a column
# print(dataset['stadium'].value_counts())

# Create the target variable: 1 if Home Goals > Away Goals, else 0
dataset["home_win"] = (dataset['Goals Home'] > dataset['Away Goals']).astype(int)

# Define predictor features. Based on offensive and possession features
features = ['home_possessions', 'away_possessions', 'home_shots', 'away_shots', 'home_on', 'away_on',
            'home_off', 'away_off', 'home_pass', 'away_pass', "home_saves", "away_saves"]

# Ensure these features are numeric in case of any data type issues
X = dataset[features].apply(pd.to_numeric, errors='coerce')
y = dataset['home_win']

# Drop rows with missing values in X or y
data = pd.concat([X, y], axis=1).dropna()
X = data[features]
y = data['home_win']

#Logistic regression
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Logistic Regression model: {accuracy:.2f}')