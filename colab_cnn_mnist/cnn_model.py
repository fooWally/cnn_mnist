import os, sys, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, metrics, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
#from tensorflow.keras.regularizers import l2  # L2 regularization


#---------------------------------------------------------
#   Build a simple CNN model and
#   Try practicing customized trainig instead of model.fit method
def build_model(input_shape, num_classes):
    #print('Input.shape =', Input(shape=input_shape).shape)
    #(num_samples, 20, 20, 1) shape is expected
    model = Sequential(
        [
            Input(shape=input_shape),
            #Input(shape=input_shape, batch_size=64),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])
    return model


@tf.function
def train_step(Xtrain, ytrain, model, loss_object, optimizer, train_loss, train_accuracy):
    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the loss
        pred = model(Xtrain, training=True)
        # sparse_categorical_crossentropy without one-hot-encode
        loss = loss_object(ytrain, pred)

    # calculate the gradients using our tape and then update the weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) #You want to apply gradients on all trainable variables, not only to the model weights

    train_loss(loss)
    train_accuracy(ytrain, pred)


@tf.function
def val_step(Xval, yval, model, loss_object, val_loss, val_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    pred = model(Xval, training=False)
    t_loss = loss_object(yval, pred)

    val_loss(t_loss)
    val_accuracy(yval, pred)


def train_epochs(train_dataset,val_dataset,model, epochs=50):
    #-------------------------------------------------------------
    #   define parameters and customize configuration
    EPOCHS = epochs # num Epochs of training
    BS = 128        # batch size, the number of images to train at one train_step
    INIT_LR = 1e-3  # initial learning rate
    optimizer      = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    loss_object    = losses.SparseCategoricalCrossentropy(name='train_loss')
    train_loss     = metrics.Mean(name='rain_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name='train_acc')
    val_loss       = metrics.Mean(name='val_loss')
    val_accuracy   = metrics.SparseCategoricalAccuracy(name='val_acc')
    history = {'loss':[], 'acc':[], 'val_loss':[],'val_acc':[]}

    for epoch in range(0, EPOCHS):
        # show the current epoch number
        sys.stdout.flush()
        epochStart = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # for all training data set : 1237 images
        for images, labels in train_dataset:
            train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)
        # for all validation data set : 200 images
        for val_images, val_labels in val_dataset:
            val_step(val_images, val_labels, model, loss_object, val_loss, val_accuracy)

        template = 'Epoch {}/{}, loss: {}, acc: {}, val_loss: {}, val_acc: {}'
        print(template.format(epoch+1, EPOCHS,
                                train_loss.result(), train_accuracy.result()*100,
                                val_loss.result(), val_accuracy.result()*100))

        history['loss'].append(train_loss.result())
        history['acc'].append(train_accuracy.result())
        history['val_loss'].append(val_loss.result())
        history['val_acc'].append(val_accuracy.result())

        # show timing information for the epoch
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0
        print("took {:.4} minutes".format(elapsed))

    # save weights
    model.save('mnist.h5')
    return history









