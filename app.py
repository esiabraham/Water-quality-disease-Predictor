
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

water_quality_dataset=pd.read_csv("/content/data.csv")


X = water_quality_dataset.drop("disease", axis=1)
y = water_quality_dataset["disease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the machine learning model (Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit web application
st.title("AquaSol: Water Quality Disease Outbreak Predictor")

# Display Data Summary
st.subheader("Data Summary:")
st.write(water_quality_dataset.describe())

#Distribution
plot = sns.pairplot(water_quality_dataset, hue ='disease',diag_kind="kde")
st.pyplot(plot.fig)


# Input form for user to enter pH level and temperature
ph_level = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0)

# Make predictions based on user input
user_input = [[ph_level, temperature]]
prediction = model.predict(user_input)[0]

st.write(f"Predicted Disease: {prediction}")

# Model Evaluation
st.subheader("Model Evaluation:")
st.write("Accuracy:", accuracy_score(y_test, y_pred))


st.title("Disease Countplot")

# Horizontal countplot of diseases with count values displayed
fig, ax = plt.subplots()
sns.countplot(y="disease", data=water_quality_dataset, ax=ax)

# Add count values as annotations
for p in ax.patches:
    ax.annotate(f'{p.get_width()}', (p.get_x() + p.get_width() + 0.1, p.get_y() + p.get_height() / 2), ha='left', va='center')

ax.set_title("Count of Diseases")
ax.set_xlabel("Count")
ax.set_ylabel("Disease")

# Display the plot in Streamlit
st.pyplot(fig)