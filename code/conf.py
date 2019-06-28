import os

data_dir = 'data'
model_dir = 'model'

train_csv = os.path.join(data_dir, 'mnist_train.csv')
test_csv = os.path.join(data_dir, 'mnist_test.csv')

cnn_model_json = os.path.join(model_dir, 'model.json')
cnn_model_weight = os.path.join(model_dir, 'model.h5')
