import numpy as np
import sys
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

import logging
logging.getLogger('tensorflow').disabled = True

try: #python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except: pass

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython split_train_test.py TEST_RATIO SEED\n')
    sys.stderr.write('\t\tEPOCHS - No. of times dataset is passed . Example: 2\n')
    sys.stderr.write('\t\tBATCH SIZE - divide dataset into n parts. Example: 20170423\n')
    sys.exit(1)

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])

# Defining CNN model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')
X_val = np.load('data/X_val.npy') 
Y_val = np.load('data/Y_val.npy')

# Without data augmentation i obtained an accuracy of 0.98114
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data = (X_val, Y_val), verbose = 2)

scores = model.evaluate(X_val, Y_val)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model/model.h5")
sys.stderr.write("Saved model to disk")
