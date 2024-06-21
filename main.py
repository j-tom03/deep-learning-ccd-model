from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data - skipping 1 row to avoid labels
data = pd.read_excel(r"CCD.xls", skiprows=1)

# One Hot Encoding on 'SEX' 'MARRIAGE' 'EDUCATION' columns
data = pd.get_dummies(data, columns=['SEX'], prefix='SEX')
data = pd.get_dummies(data, columns=['MARRIAGE'], prefix='MARRIAGE')
data = pd.get_dummies(data, columns=['EDUCATION'], prefix='EDUCATION')

# Split data to X and Y
x = data.drop("default payment next month", axis=1)
y = data["default payment next month"]

# Define a fold counter
fold = 0

# Create a StratifiedKFold object for K = 5
skfold = StratifiedKFold(n_splits=5)

# Create lists to capture the history of the model
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for train, test in skfold.split(x, y):
    fold += 1
    print("processing fold: "+str(fold))

    # Split the data into train and test sets
    x_train = x.iloc[train]
    x_test = x.iloc[test]
    y_train= y.iloc[train]
    y_test = y.iloc[test]

    # Convert the data to float32 type
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

    # Apply StandardScaler to normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # Create the model with Sequential class
    model = Sequential()

    # Add layers to the model
    model.add(Input((33,))) # 33 is the number of features
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.5))

    # Add the output layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=80, batch_size=120, validation_split=0.2)

    # Evaluate the model against test sets
    loss, accuracy = model.evaluate(x_test, y_test)

    # Print the test loss and accuracy for the fold
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')

    # Capture the history of the model for plots
    val_losses.append(history.history['val_loss'])
    val_accuracies.append(history.history['val_accuracy'])
    train_losses.append(history.history['loss'])
    train_accuracies.append(history.history['accuracy'])

# Calculating the mean of the losses and accuracies
mean_val_loss = np.mean(np.array(val_losses), axis=0)
mean_train_loss = np.mean(np.array(train_losses), axis=0)
mean_val_accuracy = np.mean(np.array(val_accuracies), axis=0)
mean_train_accuracy = np.mean(np.array(train_accuracies), axis=0)

# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(mean_train_loss, label='Training Loss')
plt.plot(mean_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(mean_train_accuracy, label='Training Accuracy')
plt.plot(mean_val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()