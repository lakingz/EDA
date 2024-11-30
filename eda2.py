import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Load the dataset from the raw GitHub URL
url = "https://raw.githubusercontent.com/lakingz/EDA/main/data/generated_sales_data.csv"
df=pd.DataFrame()
df = pd.read_csv(url)

len(df)

df = df.drop_duplicates()

df = df.dropna(subset=['Region'])

# Drop the 'PaymentMethod' column
try:
    df = df.drop(columns=['Payment Method'])
except KeyError:
    print("'PaymentMethod' column does not exist.")

# Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Feature Engineering: Time-based features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()  # Day of the week
df["IsWeekend"] = df["Weekday"].isin(["Saturday", "Sunday"])  # Boolean for weekend

# Extract numbers from 'Product Name'
df['Product Number'] = df['Product Name'].apply(lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else None)

# Apply one-hot encoding to 'Product Category'
one_hot_encoded = pd.get_dummies(df['Product Category'], prefix='Category')

# Add the one-hot encoded columns back to the original DataFrame
df = pd.concat([df, one_hot_encoded], axis=1)
df