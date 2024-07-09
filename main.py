import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import keras_tuner as kt
from sklearn.model_selection import train_test_split

tf.random.set_seed(1)

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


directory = "data/all_data"
all_data_labels = ["adenocarcinoma","large.cell.carcinoma","normal","squamous.cell.carcinoma"]
all_data = tf.keras.preprocessing.image_dataset_from_directory(directory = directory, labels = "inferred", class_names = all_data_labels,batch_size=None, shuffle = True, seed = 1421)
all_data.shuffle(buffer_size = 1000, seed = 8732)

all_data_list = list(all_data.as_numpy_iterator())
images,labels = zip(*all_data_list)
images = np.stack(images)
labels = np.stack(labels)

train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

def make_tf_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.batch(batch_size=32,drop_remainder=False)
    
training_resplit = make_tf_dataset(train_images, train_labels)
validation_resplit = make_tf_dataset(val_images, val_labels)
testing_resplit = make_tf_dataset(test_images, test_labels)

# Optional: Shuffle the datasets
training_resplit = training_resplit.shuffle(buffer_size=22, seed=34, reshuffle_each_iteration=True)
validation_resplit = validation_resplit.shuffle(buffer_size=5, seed=432, reshuffle_each_iteration=True)
testing_resplit = testing_resplit.shuffle(buffer_size=5, seed=53, reshuffle_each_iteration=True)

tuner_resplit = kt.Hyperband(hypertuned_mod_builder, #define hyperband tuner
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=4,
                     directory='hyper_tuning_resplit',
                     project_name='chest_cancer_proj')

stop_resplit = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) 

tuner_resplit.search(training_resplit, validation_data=validation_resplit, epochs=50, callbacks=[stop_resplit])

best_hps=tuner_resplit.get_best_hyperparameters(num_trials=1)[0]
mod_best = tuner_resplit.hypermodel.build(best_hps)
mod_best.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
checkpoint_path = "training_1/mod_best_resplit.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=1,mode = "min")
history_best = mod_best.fit(training_resplit,validation_data = validation_resplit, epochs=100,callbacks = [cp_callback])

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
loss, acc = mod_best.evaluate(testing_resplit, verbose=2)
print("Best model, testing accuracy: {:5.2f}%".format(100 * acc))


