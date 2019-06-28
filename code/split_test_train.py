#preprocessing
import sys
import conf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

INPUT = conf.train_csv

try: #python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except: pass

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython split_train_test.py TEST_RATIO SEED\n')
    sys.stderr.write('\t\tTEST_RATIO - train set ratio (double). Example: 0.3\n')
    sys.stderr.write('\t\tSEED - random state (integer). Example: 20170423\n')
    sys.exit(1)

test_ratio = float(sys.argv[1])
seed = int(sys.argv[2])

# Load the data
train = pd.read_csv(INPUT)

sys.stderr.write('Training data size is {}\n'.format(train.shape[0]))

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

# Normalize the data
X_train = X_train / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = test_ratio, random_state=seed)

np.save('data/X_train.npy', X_train)
np.save('data/Y_train.npy', Y_train)
np.save('data/X_val.npy', X_val)
np.save('data/Y_val.npy', Y_val)



#dvc run -d data/mnist_train.csv -d preprocess.py -d conf.py -o data/X_train.npy -o data/Y_train.npy -o data/X_val.npy -o data/Y_val.npy python preprocess.py 0.33 2