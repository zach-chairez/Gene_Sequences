In this document, we'll cover some common approaches to the binary classification problem.  Our  main is determining if a reed belongs to A1 or D1 in Cotton (G.Hirsutum)


# Section 1: Logistic Regression
- Logistic regression is (specifically) a binary classification machine learning approach where the inputs are numerical and the outputs are binary (usually 0 or 1).

```python
# Here we'll import necessary packages.
from sklearn.linear_model import LogisticRegression as lr
import pandas as pd
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

# Section 2:  k-Nearest Neighbors (kNN)




# Section 3:  Support Vector Machines (SVM)
