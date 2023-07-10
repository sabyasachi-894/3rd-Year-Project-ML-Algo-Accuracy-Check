import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Step 2: Load and preprocess data
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Project 3rd Year\dataset.csv')
# Perform any necessary preprocessing steps here

# Split the data into input features and target variable
X = data.drop('Label', axis=1)
y = data['Label']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocess the data by scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Choose a model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=164, activation='softmax'))

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 5: Make predictions
y_pred_probs = model.predict(X_test)
y_pred = np.round(y_pred_probs).astype(int).flatten()

# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#Accuracy: 0.007662835249042145