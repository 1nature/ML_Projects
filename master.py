import kagglehub
import numpy as np
import pandas as pd
import os


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
dataset.info()


