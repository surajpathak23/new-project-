# Importing libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Sample dataset
# file_path = "C:\Users\PRATI\Desktop\Xplore Training\Python Introduce Overview\Day_17 - 24th_July\sample_customer_data.csv"  # Update with your actual file path
df = pd.read_csv('customer_clv_data.csv')  # Must contain 'features' & 'target' columns
# print(df.head())


# Splitting data
# Drop ID column if not useful
# data cleaning

# df['CustomerID'] = df['CustomerID'].str.replace("C",'')
df = df.drop(['CustomerID'], axis=1, errors='ignore')

# Convert all non-numeric columns using get_dummies()
X = df.drop("target", axis=1)
X = pd.get_dummies(X)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Training model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)



print("\nTop Factors Driving CLV:\n")
print(feature_importance.head(5))

# Recommendation Logic
def business_insight(feature, value):
    if feature == "purchase_frequency" and value > 10:
        return "Encourage loyalty upgrades"
    elif feature == "avg_order_value" and value < 100:

        return "Bundle products to raise order value"
    else:

        return "Maintain engagement strategy"

# Example usage
import streamlit as st
import matplotlib.pyplot as plt

st.title("         CLV Evaluation Dashboard      ")

# Interactive Visualization

st.subheader("Feature Importance")
st.dataframe(feature_importance)
st.subheader("Model Performance")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))



st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ax.matshow(confusion_matrix(y_test, y_pred), cmap="Blues")
st.pyplot(fig)


# Create BarChart for Feature Importance using Streamlit

import matplotlib.pyplot as plt

# Feature Importance Bar Chart using Matplotlib
st.subheader("Feature Importance Bar Chart")
fig, ax = plt.subplots()
ax.barh(feature_importance["Feature"], feature_importance["Importance"], color='skyblue')
ax.invert_yaxis()  # Highest importance on top
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance - Random Forest")

st.pyplot(fig)
