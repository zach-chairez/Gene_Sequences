- In this document, we'll cover some common approaches to the binary classification problem.  Our objective is to determine if a reed belongs to A1 or D1 in Cotton (G.Hirsutum).
- The functions ```read_fasta_file``` and ```change_bases_to_int``` can be found in the folder ```python_functions```.



# Section 1: Logistic Regression
- Logistic regression is (specifically) a binary classification machine learning approach where the inputs are numerical and the outputs are binary (usually 0 or 1).

```python
# Here we'll import necessary packages.
from sklearn.linear_model import LogisticRegression as lr
import numpy as np
```

Next, we'll upload a fasta file which contains the cotton info.  There are eight (8) different G.Hirsutum files in ```/project/90daydata/gbru_sugarcane_seq/Zach/Cotton``` which were originally downloaded from CottonGen
```python
cotton_path = '/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/insert_cotton_file_of_interest_here.fa'
```

Then, we'll upload the fasta file and select the chromosomes A1 and D1
```python
_,sequences = read_fasta_file(cotton_path)
a1 = sequences[0]; a1 = a1.upper()
d1 = sequences[13]; d1 = d1.upper()
a1_ints = change_bases_to_int(a1)
d1_ints = change_bases_to_int(d1)
```
The module ```upper()``` turns all the lwoercase letters uppercase.  We'll then turn the bases into integers.  

```python
train = []; num_train = 100000
read_length = 2000
for i in range(0,num_train):
  n1 = np.random.randint(0,len(a1_ints)-read_length-1)
  train.append(a1_ints[n1:n1+read_length])
for i in range(0,num_train):
  n1 = np.random.randint(0,len(d1_ints)-read_length-1)
  train.append(d1_ints[n1:n1+read_length])
```

There are multiple notes to make here:
1.  ```num_train``` can be assigned to any number of training sequences you like (in general, the more the better).
2.  ```read_length``` here is set to a value of 2000 (although HiFi reads tend to be much larger).  The idea is that we'll train the model here, then once we have a read of interest we want to test, we can randomly sample read lengths of length ```read_length```, then test where it came from.  We look at the majority of the reads classified as the label (either A1 or D1).

Now, we need to organize our data for input.
```python
train = np.resize(train,(num_train*2,read_length))
train_labels = np.concatenate([np.zeros(num_train),np.ones(num_train)],axis=0)
train_labels = train_labels.reshape(len(train_labels),1)
```

The chromosome A1 is assigned as class 0, and D1 is class 1.  Now, we'll train and test.

```python
model_lr = lr(random_state=0).fit(train,train_labels)

test = []; num_test = 100
for i in range(0,num_test):
  n1 = np.random.randint(0,len(a1_ints)-read_length-1)
  test.append(a1_ints[n1:n1+read_length])
for i in range(0,num_test):
  n1 = np.random.randint(0,len(d1_ints)-read_length-1)
  test.append(d1_ints[n1:n1+read_length])

test = np.resize(test,(num_test*2,read_length))
test_labels = np.concatenate([np.zeros(num_test),np.ones(num_test)],axis=0)

predictions = model_lr.predict(test)
check_accuracy = predictions == test_labels
accuracy_lr = sum(check_accuracy)/len(predictions)
```

# Section 2:  k-Nearest Neighbors (kNN)
*Refer to Section 1 for preparing the training and testing data*

```python
import sklearn
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt
import seaborn as sns

num_neighbors = 3
model_knn = knn(n_neighbors = num_neighbors)
model_knn.fit(train,train_labels)

predictions = model_knn.predict(test)
check_accuracy = predictions == test_labels
accuracy_knn = sum(check_accuracy)/len(predictions)
```

Notes:
1.  The value of ```num_neighbors``` should  be tweaked.  We can always implement a cross validation algorithm to find a best value.  

```python
max_k_value = 20
k_values = [i for i in range(1,max_k_value)]
scores = []
cross_val = 5

for k_val in k_values:
  model_knn = knn(n_neighbors = k_val)
  score = sklearn.model_selection.cross_val_score(model_knn,train,train_labels,cv = cross_val)
  scores.append(np.mean(score))
```

The values in the list ```scores``` are the accuracies of the model given ```k_val```.  We can plot the results to display which one works best (this part is optional).

```python
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel('Values of k')
plt.ylabel("Accuracy using Cross Validation")
```

We can then pick the value of k which works best, then train and test our model.

```python
best_index = np.argmax(scores)
best_k = k_values[best_index]

model_knn = knn(n_neighbors = best_k)
model_knn.fit(train,train_labels)

predictions = model_knn.predict(test)
check_accuracy = predictions == test_labels
accuracy_knn = sum(check_accuracy)/len(predictions)
```


# Section 3:  Support Vector Machines (SVM)
*Refer to Section 1 for preparing the training and testing data*

```python
from sklearn import svm

# Here, you can try ```svm.SVC(), svm.LinearSVC()```, or ```svm.SGDClassifier()```.  I went with the first here.
model_svm = svm.SVC()
model_svm.fit(train,train_labels)

model_svm.predict(test)
check_accuracy = predictions == test_labels
accuracy_svm = sum(check_accuracy)/len(predictions)
```
