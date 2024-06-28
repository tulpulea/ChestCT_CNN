import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import keras_tuner as kt

tf.random.set_seed(1)

directory = "data/train"
class_names = ["normal","adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib","large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa","squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"]
class_labels = ["normal","adenocarcinoma","large cell carcinoma", "squamous cell carcinoma"]
training_data = tf.keras.preprocessing.image_dataset_from_directory(directory = directory, labels = "inferred", class_names = class_names)
test_labels = ["normal","adenocarcinoma","large.cell.carcinoma","squamous.cell.carcinoma"]
testing_data = tf.keras.preprocessing.image_dataset_from_directory(directory = "data/test", labels = "inferred", class_names = test_labels)
val_labels = ["normal","adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib","large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa","squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"]
val_data = tf.keras.preprocessing.image_dataset_from_directory(directory = "data/valid", labels = "inferred", class_names = val_labels)

normalize = tf.keras.Sequential([
  layers.Rescaling(1./255,input_shape=(256, 256, 3)),
])

augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.1),
    layers.RandomCrop(240,240),
    layers.RandomZoom((-0.1,0.1)),
])

model = models.Sequential([
    tf.keras.Input(shape=(256,256,3),dtype="uint8"),
])

model.add(normalize)
model.add(augmentation)

model.add(layers.Conv2D(16,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(16, activation="relu"))

model.add(layers.Dense(16,activation="relu"))
model.add(layers.Dense(4))
model.add(layers.Softmax())
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


checkpoint_path = "training_1/mod.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=1,mode = "min")

epochs=10
history = model.fit(
  training_data,
  validation_data=val_data,
  epochs=epochs,
  callbacks = [cp_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
plt.show()

model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(testing_data, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

def hypertuned_mod_builder(hp):
    # create the model
    mod = models.Sequential()
    #input
    mod.add(tf.keras.Input(shape=(256,256,3), dtype= "uint8"))
    # standardize
    mod.add(layers.Rescaling(1./255,input_shape=(256, 256, 3)))
    # data augmentation steps - no need for hypertuning
    mod.add(layers.RandomRotation(0.1))
    mod.add(layers.RandomContrast(0.1))
    mod.add(layers.RandomCrop(240,240))
    mod.add(layers.RandomZoom((-0.1,0.1)))

    #first convolutional and pooling layers - tune number of filters and filter size
    hp_nfilters_1 = hp.Int("filters1",min_value = 32,max_value = 128,step = 32)
    hp_filtersize_1 = hp.Choice("kernel_size1",values = [3,4,5])
    mod.add(layers.Conv2D(filters=hp_nfilters_1,kernel_size=hp_filtersize_1,activation="relu"))
    mod.add(layers.MaxPooling2D((2,2)))

    #second convolutional and pooling layers - tune number of filters and filter size
    hp_nfilters_2 = hp.Int("filter2",min_value = 32,max_value = 128,step = 32)
    hp_filtersize_2 = hp.Choice("kernel_size2",values = [3,4,5])
    mod.add(layers.Conv2D(filters=hp_nfilters_2,kernel_size=hp_filtersize_2,activation="relu"))
    mod.add(layers.MaxPooling2D((2,2)))

    #third convolutional and pooling layers - tune number of filters and filter size
    hp_nfilters_3 = hp.Int("filters3",min_value = 32,max_value = 128,step = 32)
    hp_filtersize_3 = hp.Choice("kernel_size3",values = [3,4,5])
    mod.add(layers.Conv2D(filters=hp_nfilters_3,kernel_size=hp_filtersize_3,activation="relu"))
    mod.add(layers.MaxPooling2D((2,2)))

    # flatten and dense layer
    mod.add(layers.Flatten())
    hp_units_4 = hp.Int("units1",min_value = 10,max_value = 100,step = 10)
    mod.add(layers.Dense(units=hp_units_4, activation="relu"))

    #last dense layer before output
    hp_units_5 = hp.Int("units2",min_value = 10,max_value = 100,step = 10)
    mod.add(layers.Dense(units=hp_units_5,activation="relu"))
    mod.add(layers.Dense(4))
    mod.add(layers.Softmax())

    # compile
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    mod.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    return mod

tuner = kt.Hyperband(hypertuned_mod_builder,
                     objective='val_accuracy',
                     max_epochs=50, 
                     factor=4,
                     directory='hyper_tuning',
                     project_name='chest_cancer_proj')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner.search(training_data, validation_data=val_data, epochs=50, callbacks=[stop_early]) 

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)

mod_best = tuner.hypermodel.build(best_hps)
mod_best.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

checkpoint_path = "training_1/mod_best.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=1,mode = "min")

history_best = mod_best.fit(training_data,validation_data = val_data, epochs=100,callbacks = [cp_callback]) 
acc = history_best.history['accuracy']
val_acc = history_best.history['val_accuracy']

loss = history_best.history['loss']
val_loss = history_best.history['val_loss']

epochs_range = range(100) 

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
plt.show()

mod_best.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = mod_best.evaluate(testing_data, verbose=2)
print("Best model, testing accuracy: {:5.2f}%".format(100 * acc))


