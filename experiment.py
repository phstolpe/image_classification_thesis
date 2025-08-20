import os
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL.Image
import pathlib
import numpy as np
from tensorflow._api.v2.data import AUTOTUNE
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential




# func som tar split
def split_dataset(dataset, test_percentage):
    image_count = dataset.cardinality().numpy()
    test_size = int(image_count * test_percentage)
    train_ds = dataset.skip(test_size)
    test_ds = dataset.take(test_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    test_ds = configure_for_performance(test_ds)
    train_ds = configure_for_performance(train_ds)

    return train_ds, test_ds


#build model
def build_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
    input_shape =(img_height ,img_width,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    return model

#train model
def train_model(model, train_ds, epochs):
    model.fit(
        train_ds,
        epochs=epochs
    ) 

#test model
def test_model(model, test_ds):
    test_acc = model.evaluate(test_ds, verbose=2)
    print('\n Test accuracy: ', test_acc)

def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    #Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width]) 

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

#program
data_dir = './processed_datasets/dataset1/'
second_data_dir = './processed_datasets/dataset2/'
data_dir = pathlib.Path(data_dir).with_suffix('')

img_height = 180
img_width = 180
batch_size = 32

ds = tf.data.Dataset.list_files(str(data_dir/'*/*/'))
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
train_ds, test_ds = split_dataset(ds, 0.2)
# image_batch, label_batch = next(iter(train_ds))

model = build_model()
epochs = 3 
train_model(model, train_ds, epochs)
test_model(model, test_ds)




# plt.figure(figsize=(10, 10))
# for i in range(9):
#   ax = plt.subplot(3, 3, i + 1)
#   plt.imshow(image_batch[i].numpy().astype("uint8"))
#   label = label_batch[i]
#   plt.title(class_names[label])
#   plt.axis("off")
# plt.show()

