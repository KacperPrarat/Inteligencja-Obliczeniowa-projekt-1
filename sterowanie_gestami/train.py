import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.utils import image_dataset_from_directory
from keras.models import Sequential

DATASET = "./archive"
image_size = (300, 200)
batch_size = 16

train_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="training",
    seed=285800,
    image_size=image_size,
    batch_size=batch_size
)

validation_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="validation",
    seed=285800,
    image_size=image_size,
    batch_size=batch_size
)

def create_model(input_shape):
    model = Sequential([
        Input(shape=(*input_shape, 3)),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu', kernel_initializer='he_uniform'),
        Dense(3, activation='softmax')
    ])

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(train_data, validation_data, batch_size, input_shape):
    model = create_model(input_shape)
    history = History()
    checkpoint = ModelCheckpoint('game-model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

    history = model.fit(train_data, validation_data=validation_data, epochs=15, batch_size=batch_size, callbacks=[history, checkpoint ]) #,early_stopping])
    
    cm = calculate_confusion_matrix_and_accuracy(model, validation_data)
    plot_diagnostics(history, cm)

def calculate_confusion_matrix_and_accuracy(model, validation):
  all_labels = []
  all_predictions = []
  for data_batch, labels_batch in validation:
    predictions_batch = model.predict_on_batch(data_batch)
    all_labels.extend(np.argmax(labels_batch, axis=-1))
    all_predictions.extend(predictions_batch)

  predicted_classes = np.argmax(np.array(all_predictions), axis=-1) 

  cm = confusion_matrix(all_labels, predicted_classes)

  _, test_acc = model.evaluate(validation)

  return cm, test_acc

def plot_diagnostics(history, cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    # plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', color='grey')
    plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()

train_model(train_data, validation_data, batch_size, image_size)
