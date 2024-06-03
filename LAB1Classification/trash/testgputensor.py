import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#data, load csv dataset
data = pd.read_csv('diabetes_data.csv')

# Load and preprocess the data (assuming the data is already loaded in the variable `data`)
X = data.drop('Diabetes', axis=1)
y = data['Diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to train and evaluate models
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Logistic Regression
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg_acc, log_reg_report = train_evaluate_model(log_reg, X_train, y_train, X_test, y_test)

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree_acc, tree_report = train_evaluate_model(tree, X_train, y_train, X_test, y_test)

# Random Forest
forest = RandomForestClassifier(random_state=42)
forest_acc, forest_report = train_evaluate_model(forest, X_train, y_train, X_test, y_test)

# Gradient Boosting
gbc = GradientBoostingClassifier(random_state=42)
gbc_acc, gbc_report = train_evaluate_model(gbc, X_train, y_train, X_test, y_test)

# Support Vector Machine
svc = SVC(random_state=42)
svc_acc, svc_report = train_evaluate_model(svc, X_train, y_train, X_test, y_test)

# Neural Network using TensorFlow (GPU enabled)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
nn_loss, nn_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")
nn_report = classification_report(y_test, y_pred_nn)

# Output results
results = {
    "Logistic Regression": (log_reg_acc, log_reg_report),
    "Decision Tree": (tree_acc, tree_report),
    "Random Forest": (forest_acc, forest_report),
    "Gradient Boosting": (gbc_acc, gbc_report),
    "SVM": (svc_acc, svc_report),
    "Neural Network": (nn_acc, nn_report)
}

results
