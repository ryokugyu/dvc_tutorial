# DVC Tutorial

Machine Learning projects deals with both the data and code. DVC is a tool which
provides data versioning. For more information on DVC visit [here](https://www.dvc.org).

The main purpose of this tutorial is to get a brief overview of the DVC and how
it is solving modern machine learning project issues.

## Contents of the repository


Follow these steps:

### Clone this repository
```
git clone https://github.com/ryokugyu/dvc_tutorial.git
```

After cloning the repository, change directory to `dvc_tutorial`.

### Initialize DVC repository
```
dvc init
```
After initializing the DVC repository. Let's pull the data into our machine
locally:

```
dvc push
```
Now we have both the data and code present locally in our machine. First, split
the dataset into 70-30% ratio.

### Splitting the dataset:

```
dvc run -d data/mnist_train.csv -d code/split_test_train.py -d code/conf.py -o data/X_train.npy -o data/Y_train.npy -o data/X_val.npy -o data/Y_val.npy python code/split_test_train.py 0.33 2
```

- Now, lets create a folder named `model`.

```

mkdir model
```
### compiling and training the model. also validating the performance.
### storing the model matrix also

```
dvc run -v -d  data/X_train.npy -d data/Y_train.npy -d data/X_val.npy -d data/Y_val.npy -d code/conf.py -d code/model_train.py -o model/model.json -o model/model.h5 python code/model_train.py 1 256
```

### loading the model and testing the model performance with exxternal testing dataset

 ```
 dvc run -d data/mnist_train.csv -d code/conf.py -d code/model_test.py -M data/eval.txt -f Dvcfile python code/model_test.py
 ```

DVC metric feature

```
dvc metrics show
```
