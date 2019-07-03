# DVC Tutorial


### Initialize git repository
`git init`

### Initialize DVC repository
`dvc init`

### For training the model first. splitting the training dataset into 70-30 ratio nad saving intermediate output files.

```dvc run -d data/mnist_train.csv -d code/split_test_train.py -d code/conf.py -o data/X_train.npy -o data/Y_train.npy -o data/X_val.npy -o data/Y_val.npy python code/split_test_train.py 0.33 2
```

- Create a folder named `model`.

### compiling and training the model. also validating the performance. 
### storing the model matrix also

```
dvc run -v -d  data/X_train.npy -d data/Y_train.npy -d data/X_val.npy -d data/Y_val.npy -d code/conf.py -d code/model_train.py -o model/model.json -o model/model.h5 python code/model_train.py 1 256
```

### loding the model and testing the model performance with exxternal testing dataset

 ```
 dvc run -d data/mnist_train.csv -d code/conf.py -d code/model_test.py -M data/eval.txt -f Dvcfile python code/model_test.py
 ```

DVC metric feature

```
dvc metrics show
```
