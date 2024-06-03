import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
from scipy.stats import sem
import joblib
import matplotlib.pyplot as plt

# Load the preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

# Split the data into features and target variable
X = data.drop(columns=['Diabetes'])
y = data['Diabetes']
y = y.apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Define Neural Network model
class NeuralNetworkModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
model = NeuralNetworkModel(input_dim)
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = binary_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training Loop
epochs = 10
for epoch in range(epochs):
    loss = train_step(model, X_train, y_train)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.numpy():.4f}')

# Define the evaluation function
def evaluate_model(model, X_test, y_test):
    logits = model(X_test)
    predictions = tf.round(logits)
    predictions = tf.squeeze(predictions)
    y_test_numpy = y_test.values
    predictions_numpy = predictions.numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_numpy, predictions_numpy)
    
    # Calculate precision
    precision = precision_score(y_test_numpy, predictions_numpy)
    
    # Calculate sensitivity (recall)
    recall = recall_score(y_test_numpy, predictions_numpy)
    
    # Calculate F1 score
    f1 = f1_score(y_test_numpy, predictions_numpy)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_numpy, predictions_numpy).ravel()

    # Calculate ROC AUC
    probabilities = logits.numpy()
    roc_auc = roc_auc_score(y_test_numpy, probabilities)
    
    return accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp

# Evaluate the model
accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp = evaluate_model(model, X_test, y_test)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')

# Function to perform bootstrap resampling and calculate metrics
def bootstrap_evaluation(model, X_test, y_test, n_iterations=10):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'tp': []
    }
    
    for i in range(n_iterations):
        # Resample the test dataset
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_test_resampled = X_test[indices]
        y_test_resampled = y_test.iloc[indices]
        
        # Evaluate the model on the resampled dataset
        accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp = evaluate_model(model, X_test_resampled, y_test_resampled)
        
        # Store the metrics
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['roc_auc'].append(roc_auc)
        metrics['tn'].append(tn)
        metrics['fp'].append(fp)
        metrics['fn'].append(fn)
        metrics['tp'].append(tp)
    
    # Calculate mean and SEM for each metric
    for metric in metrics:
        mean = np.mean(metrics[metric])
        sem_value = sem(metrics[metric])
        ci_lower = mean - 1.96 * sem_value
        ci_upper = mean + 1.96 * sem_value
        print(f'{metric.capitalize()} - Mean: {mean:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})')

# Perform bootstrap evaluation
bootstrap_evaluation(model, X_test, y_test)