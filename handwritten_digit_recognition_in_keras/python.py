
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

        #creating model struct

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))         #128 is neuron number,activation is default!!!
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))         #128 is neuron number,activation is default!!!
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))       #10 is neuron number
model.compile(optimizer='adam',                                     #default go to optimizer but there is lots of optimizer option
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
       
       #started to train model

model.fit(x_train, y_train, epochs=3)
                                                                    #calculate this bec this two value cannot be too close.
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#plt.imshow(x_train[0], cmap = plt.cm.binary)                       #colour map is black now.
#plt.show()                                                         #image printing func.
  
print(x_train[1])                                                   # train loss and gain
model.save('epic_num_reader.model')                                 #saving model
new_model = tf.keras.models.load_model('epic_num_reader.model')     #reloading model

predictions = new_model.predict(x_test)                             #predictions
print(predictions)
print(np.argmax(predictions[1]))                                    #writing prediction

plt.imshow(x_test[1], cmap=plt.cm.binary)                           #choosing number-letter and made it gray-scaled.
plt.show()                                                          #show image
