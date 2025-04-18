import tensorflow as tf
import matplotlib.pyplot as plt
import os, shutil, random
import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#splitting the data into train, validation and test sets
src_dir = './bananas'
target_base = 'dataset_split'
#
# splits = {'train':0.7, 'val':0.15, 'test':0.15}
# random.seed(42)
#
# for class_name in os.listdir(src_dir):
#     class_path = os.path.join(src_dir, class_name)
#     if not os.path.isdir(class_path):
#         continue
#
#     images = os.listdir(class_path)
#     random.shuffle(images)
#
#     total = len(images)
#     train_end = int(total * splits['train'])
#     val_end = train_end + int(total * splits['val'])
#
#     split_data = {
#         'train': images[:train_end],
#         'val': images[train_end:val_end],
#         'test': images[val_end:]
#     }
#
#     for split, file_list in split_data.items():
#         split_dir = os.path.join(target_base, split, class_name)
#         os.makedirs(split_dir, exist_ok=True)
#
#         for file in file_list:
#             src = os.path.join(class_path, file)
#             dst = os.path.join(split_dir, file)
#             shutil.copy(src, dst)

#paths to split datasets
train_dir = os.path.join(target_base, 'train')
val_dir = os.path.join(target_base, 'val')
test_dir = os.path.join(target_base, 'test')

train_fresh_dir = os.path.join(train_dir, 'F_Banana')
train_stale_dir = os.path.join(train_dir, 'S_Banana')

val_fresh_dir = os.path.join(val_dir, 'F_Banana')
val_stale_dir = os.path.join(val_dir, 'S_Banana')

test_fresh_dir = os.path.join(test_dir, 'F_Banana')
test_stale_dir= os.path.join(test_dir, 'S_Banana')

#number of examples in each dir
num_fresh_tr = len(os.listdir(train_fresh_dir))
num_stale_tr = len(os.listdir(train_stale_dir))

num_fresh_val = len(os.listdir(val_fresh_dir))
num_stale_val = len(os.listdir(val_stale_dir))

num_fresh_test = len(os.listdir(test_fresh_dir))
num_stale_test = len(os.listdir(test_stale_dir))

total_train = num_stale_tr + num_fresh_tr
total_val = num_stale_val + num_fresh_val
total_test = num_stale_test + num_fresh_test

print('total training stale images:', num_stale_tr)
print('total training fresh images:', num_fresh_tr)

print('total validation stale images:', num_stale_val)
print('total validation fresh images:', num_fresh_val)

print('total test stale images:', num_stale_test)
print('total test fresh images:', num_fresh_test)

#preparing the images to be fed to the model
train_image_generator = ImageDataGenerator(rescale=1./255)
val_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 100
IMG_SHAPE = 128

train_data_gen = train_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                           directory = train_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_SHAPE, IMG_SHAPE), #128, 128
                                                           class_mode = 'binary')
val_data_gen = val_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                           directory = val_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_SHAPE, IMG_SHAPE), #128, 128
                                                           class_mode = 'binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                           directory = test_dir,
                                                           shuffle = True,
                                                           target_size = (IMG_SHAPE, IMG_SHAPE), #128, 128
                                                           class_mode = 'binary')

sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.savefig('plotted_imgs')

# plotImages(sample_training_images[:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE,3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)

])
model.summary()

model.compile(optimizer= 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
EPOCHS = 25
history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(total_val/float(BATCH_SIZE)))
    )
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
