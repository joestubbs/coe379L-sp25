Hands-on CNN 
============

In this module, we work through a hands-on lab to implement different CNN architectures on a new 
dataset. 
We are given a dataset containing images of foods belonging to three categories: 
Bread, Soup and Vegetable/Fruits.
Our goal is to classify these images into their individual classes.

Step 0: Begin with getting the data on your machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a new terminal on your Jupyter and cd into /code/data folder. 
If you do not have data folder make one using ``mkdir``

.. code-block:: console  

    mkdir data
    cd data

Inside the data folder use ``wget`` to download the ``Food_Data.zip`` file. 

.. code-block:: console

    wget https://github.com/joestubbs/coe379L-sp25/raw/master/datasets/unit03/Food_CNN/Food_Data.zip


Once the ``Food_Data.zip`` file is downloaded, unzip it. Size of zip file is approximately 90MB

.. code-block:: console  

   unzip Food_Data.zip

You should see three folders inside the ``Food_Data`` directory: ``Bread``, ``Soup`` and ``Vegetable-Fruit``.

Next, create a python notebook outside the data directory. 

.. note:: 
    
    It is important that the notebook file is **outside** of the data directory 
    for rest of the steps to work.

Step 1: Loading the Data
~~~~~~~~~~~~~~~~~~~~~~~~~
a) We're going to be splitting the data into training and test sets, and we'll be creating new directories 
   for each. In order to make our code reusable, we're going to first make sure that train and test 
   directories are empty to start with. 

   We can use the ``shutil`` module from the Python standard library. It provides a higher-level interface 
   for file and directory operations such as copying, moving, deleting, etc.

   The following code ensures all directories under the following paths are empty: 

.. code-block:: python3 

    # Let's make sure these directories are clean before we start
    import shutil
    try:
        shutil.rmtree("data/Food-cnn-split/train")
        shutil.rmtree("data/Food-cnn-split/test")
    except: 
        pass


b) Lets create ``train`` and ``test`` directories for each of the three categories: Bread, Soup and 
   Vegetable-Fruits. Here we use ``pathlib`` from the ``Path`` module to ensure the directories exist.

.. code-block:: python3 

    # We have three class which contains all the data: Bread, Soup and Vegetable-Fruit
    # Let's create directories for each class in the train and test directories.
    import os 
    # ensure directories exist
    from pathlib import Path

    Path("data/Food-cnn-split/train/Bread").mkdir(parents=True, exist_ok=True)
    Path("data/Food-cnn-split/train/Soup").mkdir(parents=True, exist_ok=True)
    Path("data/Food-cnn-split/train/Vegetable-Fruit").mkdir(parents=True, exist_ok=True)

    Path("data/Food-cnn-split/test/Bread").mkdir(parents=True, exist_ok=True)
    Path("data/Food-cnn-split/test/Soup").mkdir(parents=True, exist_ok=True)
    Path("data/Food-cnn-split/test/Vegetable-Fruit").mkdir(parents=True, exist_ok=True)

c) Next we need to collect all the paths for images in each category so we can split them 
   into train and test in a ratio 80:20. 

.. code-block:: python3 

    # we need paths of images for individual classes so we can copy them in 
    # the new directories that we created above
    all_bread_file_paths = os.listdir('data/Food_Data/Bread')
    all_soup_file_paths = os.listdir('data/Food_Data/Soup')
    all_vegetable_fruit_file_paths = os.listdir('data/Food_Data/Vegetable-Fruit')

d) Now we split the image paths into train and test by randomly selecting 80% of the images 
   to put into train and 20% for test. The ``random.sample()`` function takes two arguments, a 
   list, ``l``, and an integer, ``n``, and it samples ``n`` elements from the list ``l``. 
   We also confirm that there are no overlaps between the two splits.

.. code-block:: python3 

    import random

    train_bread_paths = random.sample(all_bread_file_paths, int(len(all_bread_file_paths)*0.8))
    print("train bread image count: ", len(train_bread_paths))
    test_bread_paths = [ p for p in all_bread_file_paths if p not in train_bread_paths]
    print("test bread image count: ", len(test_bread_paths))
    # ensure no overlap:
    overlap = [p for p in train_bread_paths if p in test_bread_paths]
    print("len of overlap: ", len(overlap))

    train_soup_paths = random.sample(all_soup_file_paths, int(len(all_soup_file_paths)*0.8))
    print("train soup image count: ", len(train_soup_paths))
    test_soup_paths = [ p for p in all_soup_file_paths if p not in train_soup_paths]
    print("test soup image count: ", len(test_soup_paths))
    # ensure no overlap:
    overlap = [p for p in train_soup_paths if p in test_soup_paths]
    print("len of overlap: ", len(overlap))

    train_vegetable_fruit_paths = random.sample(all_vegetable_fruit_file_paths, int(len(all_vegetable_fruit_file_paths)*0.8))
    print("train vegetable fruit image count: ", len(train_vegetable_fruit_paths))
    test_vegetable_fruit_paths = [ p for p in all_vegetable_fruit_file_paths if p not in train_vegetable_fruit_paths]
    print("test vegetable fruitimage count: ", len(test_vegetable_fruit_paths))
    # ensure no overlap:
    overlap = [p for p in train_bread_paths if p in test_bread_paths]
    print("len of overlap: ", len(overlap))

e) Next, we actually perform the copying of files in the train and test directories

.. code-block:: python3 

    # ensure to copy the images to the directories
    import shutil
    for p in train_bread_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Bread', p), os.path.join('data/Food-cnn-split/train/Bread', p) )

    for p in test_bread_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Bread', p), os.path.join('data/Food-cnn-split/test/Bread', p) )

    for p in train_soup_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Soup', p), os.path.join('data/Food-cnn-split/train/Soup', p) )

    for p in test_soup_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Soup', p), os.path.join('data/Food-cnn-split/test/Soup', p) )

    for p in train_vegetable_fruit_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Vegetable-Fruit', p), os.path.join('data/Food-cnn-split/train/Vegetable-Fruit', p) )

    for p in test_vegetable_fruit_paths:
        shutil.copyfile(os.path.join('data/Food_Data/Vegetable-Fruit', p), os.path.join('data/Food-cnn-split/test/Vegetable-Fruit', p) )


    # check counts:
    print("Files in train/bread: ", len(os.listdir("data/Food-cnn-split/train/Bread")))
    print("Files in train/soup: ", len(os.listdir("data/Food-cnn-split/train/Soup")))
    print("Files in train/vegetable-fruit: ", len(os.listdir("data/Food-cnn-split/train/Vegetable-Fruit")))

    print("Files in test/bread: ", len(os.listdir("data/Food-cnn-split/test/Bread")))
    print("Files in test/soup: ", len(os.listdir("data/Food-cnn-split/test/Soup")))
    print("Files in test/vegetable-fruit: ", len(os.listdir("data/Food-cnn-split/test/Vegetable-Fruit")))

By the end of these steps, your train and test each should have 3 folders for Bread, Soup and Vegetable-Fruit populated.


Step 2: Data preprocessing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that we have split the image files into train and test folders, we need to perform pre-processing needed
before they can be used for training models.
The images given to us are of different sizes. We need to select a fixed target size for each 
image, so the model can be trained on them. We'll use 150x150x3.

We also need to Rescale the images by importing ``Rescaling`` from 
``tensorflow.keras.layers.experimental.preprocessing``.
``Rescaling(scale=1./255)`` is used to rescale pixel values from the typical range of [0, 255] to the 
range [0, 1]. This rescaling is often used when dealing with image data to ensure that the values are 
within a suitable range for training neural networks.

We will use the ``tf.keras.utils.image_dataset_from_directory()`` function to create a 
TensorFlow ``tf.data.Dataset`` from the image files in a directory. 
This will create a labeled dataset for us and the labels correspond to the directory that image is in.

Here is the full code: 

.. code-block:: python3 

    # import needed classes and functions 
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling

    # path to training data 
    train_data_dir = 'data/Food-cnn-split/train'

    # controls the size of the "batches" of images streamed when accessing the datasets. 
    # this is useful to control the memory usage with very large datasets
    batch_size = 32

    # target image size 
    img_height = 150
    img_width = 150

    # note that the subset parameter can take values of "training", "validation", or "both"; 
    # the value dictates which dataset is returned (we want both)
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    # rescale instance 
    rescale = Rescaling(scale=1.0/255)

    # apply the rescale to the train and validation sets 
    train_rescale_ds = train_ds.map(lambda image,label:(rescale(image),label))
    val_rescale_ds = val_ds.map(lambda image,label:(rescale(image),label))

You should see output similar to the following:

.. code-block:: console

    Found 666 files belonging to 3 classes.
    Using 533 files for training.
    Using 133 files for validation.

It is important that we do **the same** preprocessing on the test data:

.. code-block:: python3 

    # path to test data 
    test_data_dir = 'data/Food-cnn-split/test/'

    # we do not set subset=both here because we do not want the test set split
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(img_height, img_width),
    )

    # approach 1: manually rescale data --
    rescale = Rescaling(scale=1.0/255)
    test_rescale_ds = test_ds.map(lambda image,label:(rescale(image),label))

You should see output similar to the following:

.. code-block:: console

    Found 168 files belonging to 3 classes.

Now we have pre-processed datasets ``train_rescale_ds`` and ``val_rescale_ds`` and they are 
ready to be used for training the model.

A "Basic" CNN 
~~~~~~~~~~~~~~
As mentioned, we will explore several CNN architectures in this hands-on lab. 
To being, we first build a CNN with 3 alternating convolutional and pooling layers 
and 2 dense hidden layers. The output layer will have 3 classes and will use the 
*softmax* activation function. 

We've provided the basic code below, but you will need to supply some values for the ``?``
characters. 

.. code-block:: python3 

    from keras import layers
    from keras import models
    import pandas as pd 
    from keras import optimizers

    # Intializing a sequential model
    model_cnn = models.Sequential()

    # Adding first conv layer with 64 filters and kernel size 3x3 , 
    # Recall: using padding='same' ensures the output size has the same shape as the input size
    model_cnn.add(layers.Conv2D(?, (?, ?), activation='relu', padding="same", input_shape=(?,?,?)))
    
    # Adding max pooling to reduce the size of output of first conv layer
    model_cnn.add(layers.MaxPooling2D((2, 2), padding='same'))

    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model_cnn.add(layers.MaxPooling2D((2, 2), padding='same'))

    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model_cnn.add(layers.MaxPooling2D((?, ?), padding='same'))

    # flattening the output of the conv layer after max pooling to make it ready for creating dense connections
    model_cnn.add(layers.Flatten())

    # Adding a fully connected dense layer with 100 neurons    
    model_cnn.add(layers.Dense(100, activation='relu'))

    # Adding a fully connected dense layer with 84 neurons    
    model_cnn.add(layers.Dense(84, activation='relu'))

    # Adding the output layer with * neurons and activation functions as softmax since this is a multi-class classification problem  
    model_cnn.add(layers.Dense(?, activation='softmax'))

    # Compile model
    # RMSprop (Root Mean Square Propagation) is commonly used in training deep neural networks.
    model_cnn.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generating the summary of the model
    model_cnn.summary()

Let's fit the model and run it for 20 epochs.

.. code-block:: python3 

    # fit the model from image generator
    history = model_cnn.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds
    )

How did the model perform on the training and validation sets? Do you think it is overfit or underfit?

.. code-block:: console

    . . .
    Epoch 15/20
    17/17 [==============================] - 7s 378ms/step - loss: 0.5828 - accuracy: 0.7692 - val_loss: 0.7461 - val_accuracy: 0.6090
    Epoch 16/20
    17/17 [==============================] - 7s 371ms/step - loss: 0.6245 - accuracy: 0.7430 - val_loss: 0.7134 - val_accuracy: 0.6842
    Epoch 17/20
    17/17 [==============================] - 7s 376ms/step - loss: 0.5892 - accuracy: 0.7317 - val_loss: 0.7261 - val_accuracy: 0.6617
    Epoch 18/20
    17/17 [==============================] - 7s 372ms/step - loss: 0.5566 - accuracy: 0.7711 - val_loss: 0.7052 - val_accuracy: 0.6466
    Epoch 19/20
    17/17 [==============================] - 7s 376ms/step - loss: 0.5503 - accuracy: 0.7561 - val_loss: 0.8316 - val_accuracy: 0.5714
    Epoch 20/20
    17/17 [==============================] - 7s 384ms/step - loss: 0.5493 - accuracy: 0.7805 - val_loss: 0.7026 - val_accuracy: 0.6617    

Finally, let's compute the accuracy of our model on the test set. Recall that we use ``evaluate()``
for that:

.. code-block:: python3 

    test_loss, test_accuracy = model_cnn.evaluate(test_rescale_ds, verbose=0)
    test_accuracy

When I ran the code, I obtained a validation accuracy and test accuracy of about 66%. 

LeNet-5 
~~~~~~~~~~

We saw that LeNet-5 is a shallow network and has 2 alternating convolutional and pooling layers.
Let's try to train the LeNet-5 model on our training data. Again, we provide most of the code 
below, but you will need to fill in some missing portions. 

.. code-block:: python3 


    from keras import layers
    from keras import models
    import pandas as pd 

    model_lenet5 = models.Sequential()
        
    # Layer 1: Convolutional layer with 6 filters of size 3x3, followed by average pooling
    model_lenet5.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(150,150,3)))
    model_lenet5.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # Layer 2: Convolutional layer with 16 filters of size 3x3, followed by average pooling
    model_lenet5.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model_lenet5.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # Flatten the feature maps to feed into fully connected layers
    ``which layer will you add here``

    # Layer 3: Fully connected layer with 120 neurons
    model_lenet5.add(layers.Dense(120, activation='relu'))

    # Layer 4: Fully connected layer with 84 neurons
    model_lenet5.add(layers.Dense(84, activation='relu'))

    # Output layer: Fully connected layer with num_classes neurons (e.g., 3 )
    model_lenet5.add(layers.Dense(3, activation='softmax'))

    # Compile model
    model_lenet5.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generating the summary of the model
    model_lenet5.summary()


Let's fit the model and run 20 epochs

.. code-block:: python3 

    # fit the model from image generator
    history = model_lenet5.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds
    )

We see even lower validation accuracy with this model. 

.. code-block:: console

    Epoch 15/20
    17/17 [==============================] - 2s 75ms/step - loss: 0.6182 - accuracy: 0.7936 - val_loss: 1.0043 - val_accuracy: 0.5263
    Epoch 16/20
    17/17 [==============================] - 2s 75ms/step - loss: 0.5875 - accuracy: 0.7730 - val_loss: 0.9242 - val_accuracy: 0.5789
    Epoch 17/20
    17/17 [==============================] - 2s 76ms/step - loss: 0.5795 - accuracy: 0.8105 - val_loss: 0.9585 - val_accuracy: 0.5338
    Epoch 18/20
    17/17 [==============================] - 2s 75ms/step - loss: 0.5275 - accuracy: 0.8443 - val_loss: 0.9898 - val_accuracy: 0.5414
    Epoch 19/20
    17/17 [==============================] - 2s 75ms/step - loss: 0.5223 - accuracy: 0.8105 - val_loss: 0.9238 - val_accuracy: 0.5714
    Epoch 20/20
    17/17 [==============================] - 2s 75ms/step - loss: 0.4684 - accuracy: 0.8518 - val_loss: 0.9088 - val_accuracy: 0.5564    

How does the validation accuracy change over the epochs? Could we possibly execute more epochs and 
get a better result? Consider trying 50, 100, or 200 epochs. 

You may see some signs of overfitting. There are techniques such as *data-augmentation* and 
adding ``Dropout`` layers to the model that can overcome overfitting. Time permitting, we will discuss these in a 
future lecture. 

VGG16
~~~~~~~~~~
For our last CNN, let's create a VGG16 model. For this we will use the pre-trained VGG16 model, 
but note that the pre-trained model only includes the convolutional and pooling layers (i.e., 
the feature extraction layers). We still need to add the prediction layers and fit them. 

Note that we also add a ``Dropout`` layer in the prediction layers to prevent overfitting. Perhaps
counterintuitively, the Dropout layer randomly sets a sampling of input units to 0! If you 
are interested, you can read more about it from the Keras documentation 
`here <https://keras.io/api/layers/regularization_layers/dropout/>`_.


.. code-block:: python3 
   
    # Import the pre-built VGG16 model
    from keras.applications.vgg16 import VGG16

    # Load the pre-trained VGG16 model with weights trained on ImageNet
    vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (150,150,3))
    vgg_model.summary()

    # Make all the layers of the VGG model non-trainable. i.e. freeze them. This will
    # provide a big computational savings when it comes to training. 
    for layer in vgg_model.layers:
        layer.trainable = False

    # Initialize the model
    new_model = models.Sequential()

    # Add the feature extraction layers from the VGG16 model above
    new_model.add(vgg_model)

    # Flatten the output of the VGG16 model
    new_model.add(layers.Flatten())

    # Adding a dense input layer
    new_model.add(layers.Dense(32, activation='relu'))

    # Adding dropout prevents overfitting
    new_model.add(layers.Dropout(0.2))

    # Adding second input layer
    new_model.add(layers.Dense(32, activation='relu'))

    # Adding output layer
    new_model.add(layers.Dense(3, activation='softmax'))

    # Compiling the model
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Summary of the model
    new_model.summary()

    # fit the model from image generator
    history = new_model.fit(
                train_rescale_ds,
                batch_size=32,
                epochs=20,
                validation_data=val_rescale_ds,
    )
 
It turns out that this model gives us the best validation and test accuracy to 
solve the food classification problem:

.. code-block:: console

    Epoch 15/20
    17/17 [==============================] - 19s 1s/step - loss: 0.0189 - accuracy: 0.9962 - val_loss: 0.4328 - val_accuracy: 0.9023
    Epoch 16/20
    17/17 [==============================] - 20s 1s/step - loss: 0.0173 - accuracy: 0.9981 - val_loss: 0.5381 - val_accuracy: 0.8947
    Epoch 17/20
    17/17 [==============================] - 20s 1s/step - loss: 0.0074 - accuracy: 0.9981 - val_loss: 0.4901 - val_accuracy: 0.9023
    Epoch 18/20
    17/17 [==============================] - 20s 1s/step - loss: 0.0123 - accuracy: 0.9981 - val_loss: 0.5682 - val_accuracy: 0.8947
    Epoch 19/20
    17/17 [==============================] - 20s 1s/step - loss: 0.0118 - accuracy: 0.9962 - val_loss: 0.4678 - val_accuracy: 0.9023
    Epoch 20/20
    17/17 [==============================] - 20s 1s/step - loss: 0.0102 - accuracy: 0.9962 - val_loss: 0.4684 - val_accuracy: 0.9023    

.. code-block:: python3 

    test_loss, test_accuracy = new_model.evaluate(test_rescale_ds, verbose=0)
    print(f"Loss on test: {test_loss}")
    print(f"Accuracy on test: {test_accuracy}")

    Loss on test: 0.4201587438583374
    Accuracy on test: 0.898809552192688

