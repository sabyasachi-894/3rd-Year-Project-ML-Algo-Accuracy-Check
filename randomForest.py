import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load and preprocess data
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Project 3rd Year\dataset.csv')
# Perform any necessary preprocessing steps here

# Split the data into input features and target variable
X = data.drop('Label', axis=1)
y = data['Label']

# Step 3: Choose a model
model = RandomForestClassifier()

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#Accuracy: 0.8697318007662835