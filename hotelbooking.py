# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set style
sns.set(style="whitegrid")


# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv(r"D:\Infotact project 2\hotel_bookings.csv")

# Basic Info
print(df.head())
print(df.info())
print(df.describe())


# ==============================
# 3. Data Cleaning
# ==============================

# Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Remove invalid ADR values
df = df[(df['adr'] > 0) & (df['adr'] < 1000)]


# ==============================
# 4. Feature Engineering
# ==============================

# Total nights stayed
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Create arrival date
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'] + '-' +
    df['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)

# Customer segmentation
df['segment'] = df['customer_type']
df['booking_type'] = df['market_segment']


# ==============================
# 5. Exploratory Data Analysis (EDA)
# ==============================

# Cancellation Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='is_canceled', data=df)
plt.title("Cancellation Distribution")
plt.show()

# ADR vs Cancellation
plt.figure(figsize=(6,4))
sns.boxplot(x='is_canceled', y='adr', data=df)
plt.title("ADR vs Cancellation")
plt.show()

# Lead Time vs Cancellation
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='lead_time', hue='is_canceled', bins=50)
plt.title("Lead Time Distribution by Cancellation")
plt.show()

# Correlation Matrix
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Segment vs Cancellation
plt.figure(figsize=(10,5))
sns.countplot(x='segment', hue='is_canceled', data=df)
plt.xticks(rotation=45)
plt.title("Customer Segment vs Cancellation")
plt.show()

# Lead Time vs Cancellation Probability
df.groupby('lead_time')['is_canceled'].mean().plot(figsize=(10,5))
plt.title("Lead Time vs Cancellation Probability")
plt.xlabel("Lead Time")
plt.ylabel("Cancellation Probability")
plt.show()

# Monthly ADR Trend
monthly_adr = df.groupby('arrival_date_month')['adr'].mean()

plt.figure(figsize=(10,5))
monthly_adr.plot(kind='bar')
plt.title("Monthly ADR Trend")
plt.xlabel("Month")
plt.ylabel("Average ADR")
plt.show()


# ==============================
# 6. Predictive Modeling
# ==============================

df_model = df.copy()

# Encode categorical variables
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Define features and target
X = df_model.drop('is_canceled', axis=1)
y = df_model['is_canceled']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==============================
# 7. Final Data Cleaning for Export
# ==============================

# Remove rows where adults = 0 (invalid bookings)
df = df[df['adults'] > 0]

# Drop rows with invalid dates
df = df.dropna(subset=['arrival_date'])

# Remove duplicates
df = df.drop_duplicates()

# Optional: Add useful columns for dashboard
df['revenue'] = df['adr'] * df['total_nights']
df['cancellation_status'] = df['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})

# Reset index (clean structure)
df = df.reset_index(drop=True)

print("Final Cleaned Shape:", df.shape)


# ==============================
# 8. Export Cleaned CSV
# ==============================

output_path = "D:\Infotact project 2\cleaned_hotel_bookings.csv"

df.to_csv(output_path, index=False)

print("✅ Cleaned file saved at: {output_path}")