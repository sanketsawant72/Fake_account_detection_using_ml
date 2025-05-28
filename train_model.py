import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    """Load and prepare the dataset"""
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare features and labels
    X_train = train_df.drop(columns=['fake'])
    y_train = train_df['fake']
    X_test = test_df.drop(columns=['fake'])
    y_test = test_df['fake']
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    return X_train, X_test, y_train, y_test, scaler

def build_model(input_shape):
    """Build and compile the neural network model"""
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def train_model(model, X_train, y_train):
    """Train the model with callbacks"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes))
    
    roc_auc = roc_auc_score(y_true, y_pred[:, 1])
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return y_true, y_pred, y_pred_classes

def plot_results(history, y_true, y_pred, y_pred_classes):
    """Generate and save evaluation plots"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred_classes),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

def save_artifacts(model, scaler):
    """Save model and preprocessing artifacts"""
    model.save('models/instagram_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModel artifacts saved in 'models/' directory")

def main():
    print("Starting model training...")
    
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    model = build_model(X_train.shape[1])
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, X_train, y_train)
    
    print("\nEvaluating model...")
    y_true, y_pred, y_pred_classes = evaluate_model(model, X_test, y_test)
    
    plot_results(history, y_true, y_pred, y_pred_classes)
    
    save_artifacts(model, scaler)
    
    print("\nTraining completed successfully!")
    print("Model saved as 'models/instagram_model.h5'")
    print("Evaluation plots saved in 'plots/' directory")

if __name__ == "__main__":
    main()
