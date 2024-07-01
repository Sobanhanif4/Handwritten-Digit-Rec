import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Step 1: Load and Preprocess Data (using MNIST dataset)
def load_and_preprocess_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, y_train, x_test, y_test

# Step 2: Define the Model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Step 3: Compile and Train the Model
def compile_and_train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    # Early Stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    model.fit(datagen.flow(x_train, y_train, batch_size=128),
              epochs=5,  # Reduced epochs for quicker testing
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])

    model.save('mnist_model_quick.h5')

# Step 4: Load the Model and Make Predictions
def load_model_and_predict(image_path):
    model = tf.keras.models.load_model('mnist_model_quick.h5')

    # Preprocess the input image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape to match model's input shape

    # Predict the digit
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)
    print(f'Predicted digit: {predicted_digit}')

# Main function to run the entire process
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = create_model()
    compile_and_train_model(model, x_train, y_train, x_test, y_test)
    
    # Provide the path to your image here
    image_path = '/Users/mac2016/Desktop/Handwritten_Digit_Recog_Worked/for_digit_recog.png'
    load_model_and_predict(image_path)

if __name__ == '__main__':
    main()
