import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import pickle

img_dims = 64


### Load data

def load_data(directory, _shear_range=0., _zoom_range=0., _horizontal_flip=False, _batch_size=8, _shuffle=False):
    dataGenerator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=_shear_range,
        zoom_range=_zoom_range,
        horizontal_flip=_horizontal_flip,

    )

    data = dataGenerator.flow_from_directory(
        directory,
        target_size=(img_dims, img_dims),
        batch_size=_batch_size,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=_shuffle
    )

    return data


training_set = load_data('dataset/training_set', 0.2, 0.2, True)
validation_set = load_data('dataset/validation_set')


### Visualize the Dataset

def display_images(path, directory_list):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex='col', sharey='row')
    index = 0
    for image_name in directory_list:
        input_path = os.path.join(path, image_name)
        img = plt.imread(input_path)
        ax[index].imshow(img)
        index += 1
    plt.show()


cat_directory = "dataset/training_set/cat"
dog_directory = "dataset/training_set/dog"

# display_images(cat_directory,os.listdir(cat_directory)[0:2])
# display_images(dog_directory,os.listdir(dog_directory)[0:2])

### Create an Instance of the Neural Network
### Adding the Layers to the CNN

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(img_dims, img_dims, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ]
)
model.summary()

### Compile the Model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

### Train the CNN
# history = model.fit(
#     x = training_set,
#     validation_data = validation_set,
#     epochs = 100
# )
# model.save("trained-model.keras")
# import pickle
#
# ### Save history
# with open("trained-model-history.pickle", 'wb') as file:
#     pickle.dump(history.history, file)


### Test the CNN ---- to load existing model ---> model=tf.keras.models.load_model("/usercode/trained-model")

model = tf.keras.models.load_model("trained-model.keras")
test_set = load_data('dataset/test_set', _batch_size=1, _shuffle=True)
# test_loss, test_acc = model.evaluate(test_set)
# print(f'Test Accuracy: {test_acc}')

## Visualize the Metrics + Load history

# with open("trained-model-history.pickle", 'rb') as file:
#     history = pickle.load(file)
#
# plt.plot(history['accuracy'], label = 'Training accuracy', color = 'r')
# plt.plot(history['val_accuracy'], label = 'Validation accuracy', color = 'b')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# plt.plot(history['loss'], label = 'Training loss', color = 'r')
# plt.plot(history['val_loss'], label = 'Validation loss', color = 'b')
# plt.title('Training and validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


### Plot the Confusion Matrix
y_pred = model.predict(test_set, verbose=0)
y_pred = np.round(y_pred)
true_vals = test_set.classes

confusion_mtx = tf.math.confusion_matrix(true_vals, y_pred)
sns.heatmap(confusion_mtx,
            cmap=plt.cm.Blues,
            annot=True,
            fmt='d',
            xticklabels=['Cat', 'Dog'],
            yticklabels=['Cat', 'Dog'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

