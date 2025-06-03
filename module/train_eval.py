import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig("visualizations/training_plot.png")
    plt.show()

def evaluate_model(model, val_gen):
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Classification Report')
    print(classification_report(val_gen.classes, y_pred, target_names=val_gen.class_indices))

    cm = confusion_matrix(val_gen.classes, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_gen.class_indices, yticklabels=val_gen.class_indices)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("visualizations/confusion_matrix.png")
    plt.show()
    print("Confusion Matrix:")
    print(cm)