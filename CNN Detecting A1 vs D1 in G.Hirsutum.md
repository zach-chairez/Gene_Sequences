- In this file, we'll walk through the steps taken to create the convolutional neural network (CNN) which is used to detect the origin of subreads to either A1 or D1 of G.Hirsutum.

We'll start by loading all the necessary packages:
```python
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
import random
from itertools import product
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GroupNormalization, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
```

The function ```python read_fasta_file ``` can be found in the folder python_functions.  It's used to load a fasta file path, then outputs the headers and sequences.  We'll first load the following:

```python
# read_fasta_file reads the file path, then outputs the headers and sequences.
h, seq = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton1.fa')

# Here, we're assigning the chromosomes A1 and D1 accordingly.
# The step .upper() takes all the lowercase letters (a,c,t,g) and capitalizes them.
a1 = seq[0]; a1 = a1.upper()
d1 = seq[13]; d1 = d1.upper()
```

