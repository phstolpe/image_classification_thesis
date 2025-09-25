import os
import tensorflow as tf
import pathlib
import numpy as np
import gc
from tensorflow._api.v2.data import AUTOTUNE
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential



#splits a dataset based on argument test percentage
def split_dataset(dataset, test_percentage, seed):
    image_count = dataset.cardinality().numpy()
    test_size = int(image_count * test_percentage)
    #shuffle with random seed
    dataset = dataset.shuffle(buffer_size=image_count, seed = seed)

    train_ds = dataset.skip(test_size)
    test_ds = dataset.take(test_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)

    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds)


    return train_ds, test_ds


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
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def build_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',
    input_shape =(img_height ,img_width,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation ='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    return model

def run(test_size: float, path_to_ds: str, seed: int): 
    path_to_ds = pathlib.Path(path_to_ds).with_suffix('')
    global class_names  
    class_names = np.array(sorted([item.name for item in path_to_ds.glob('*') if item.name != "LICENSE.txt"]))
    print("Number of images in dataset: ", len(list(path_to_ds.glob('*/*'))))
    ds = tf.data.Dataset.list_files(str(path_to_ds/'*/*'))

    #splits and prepares images
    train_ds, test_ds = split_dataset(ds, test_size, seed)
    model = build_model()
    model.fit(train_ds, epochs=epochs) 
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    data = {
            "accuracy" : test_acc
    }
    #clean up and free memory
    keras.backend.clear_session()
    del model
    gc.collect()

    return data

#config variables
img_height = 128 
img_width = 128 
batch_size = 32 
epochs = 15
class_names = None
