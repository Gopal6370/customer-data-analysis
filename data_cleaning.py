# Data Immersion & Wrangling
# Fully Runnable Project Code

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
print("Loading Dataset...")

df = pd.read_csv("raw_dataset_dirty.csv")

print("\nFirst 5 Rows:")
print(df.head())

# Step 2: Dataset Information

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# Step 3: Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Visualize Missing Values
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.savefig("missing_values.png")
plt.show()

# Step 4: Handle Missing Values

# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Purchase_Amount with median
df['Purchase_Amount'] = df['Purchase_Amount'].fillna(
    df['Purchase_Amount'].median()
)

# Step 5: Remove Duplicate Rows

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows Found: {duplicates}")

df = df.drop_duplicates()

# Step 6: Standardize Text Data

df['Name'] = df['Name'].str.title()
df['City'] = df['City'].str.title()


# Step 7: Convert Date Format

df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])


# Step 8: Feature Engineering

# Create Age Category
def age_category(age):
    if age < 25:
        return "Young"
    elif age < 40:
        return "Adult"
    else:
        return "Senior"

df['Age_Group'] = df['Age'].apply(age_category)


# Step 9: Statistical Summary

print("\nStatistical Summary:")
print(df.describe())

# Step 10: Correlation Heatmap

plt.figure(figsize=(6,4))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Step 11: Save Cleaned Dataset

df.to_csv("cleaned_dataset.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned dataset saved as 'cleaned_dataset.csv'")