import sys
import pandas as pd
import conf
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.optimizers import RMSprop
from keras.models import model_from_json

import logging
logging.getLogger('tensorflow').disabled = True

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

TEST_INPUT = conf.test_csv
CNN_JSON = conf.cnn_model_json
CNN_WEIGHT = conf.cnn_model_weight
metrics_file = 'data/eval.txt'

# load json and create model
json_file = open(CNN_JSON, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(CNN_WEIGHT)
print("Loaded model from disk")


test = pd.read_csv(TEST_INPUT)
Y_test = test["label"]

# Drop 'label' column
X_test = test.drop(labels = ["label"],axis = 1) 

X_test = X_test / 255.0
X_test = X_test.values.reshape(-1,28,28,1)
Y_test = to_categorical(Y_test, num_classes = 10)

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

score = loaded_model.evaluate(X_test, Y_test, verbose=0)

print("Accuracy: %.2f%%" % (score[1]*100))

with open(metrics_file, 'w') as fd:
    fd.write('AUC: {:4f}\n'.format(score[1]))

