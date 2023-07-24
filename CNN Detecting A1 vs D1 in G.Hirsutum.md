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

The function ```read_fasta_file``` can be found in the folder python_functions.  It's used to load a fasta file path, then outputs the headers and sequences.  We'll first load the following:

```python
# read_fasta_file reads the file path, then outputs the headers and sequences.
h, seq = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton1.fa')

# Here, we're assigning the chromosomes A1 and D1 accordingly.
# The step .upper() takes all the lowercase letters (a,c,t,g) and capitalizes them.
a1 = seq[0]; a1 = a1.upper()
d1 = seq[13]; d1 = d1.upper()
```
Next, we'll define two functions:  
- 1. ```find_kmers``` 
- 2.  ```subsequence_all_kmers```

The first function creates a list of all k-mers present in a sequence, while the second function generates subsequences of some length n, ensuring that all k-mers established from the first function are present in the generated subsequences.  This is important as it's necessary to create a "dictionary of words (k-mers)" for our network.  We're going to approach this problem from a natural language processing (NLP) point of view, treating each k-mer as a word in a sequence (or sentence).  

```python

# Find different k-mers in a sequence
def find_kmers(base_string,k):
    unique_permutations = set()
    # Generate all possible 4-mer permutations
    permutations = [''.join(p) for p in product('ACTGN', repeat=k)]
    # Iterate over the sequence
    for i in range(len(base_string) - k-1):
        subsequence = base_string[i:i + k]
        # Check if subsequence contains valid bases
        if all(base in 'ACTGN' for base in subsequence):
            unique_permutations.add(subsequence)
    return unique_permutations
```
```python
# Generate subsequences based on k-mer
def subsequence_all_kmers(base_string, k, n):
    kmers = find_kmers(base_string, k)
    final_subsequences = []
    while len(kmers) > 0:
        start_index = random.randint(0, len(base_string) - n)
        subsequence = base_string[start_index:start_index + n]
        if any(kmer in subsequence for kmer in kmers):
            final_subsequences.append(subsequence)
            kmers -= {kmer for kmer in kmers if kmer in subsequence}
    return final_subsequences

```

Now we'll generate subsequences to create sentences from to build our dictionary of words for our network.
```python
k_mers = 3
sub_len = 41
subsequences_a1 = subsequence_all_kmers(a1,k_mers,sub_len);
subsequences_d1 = subsequence_all_kmers(d1,3,41);
```

The sequences ```subsequences_a1``` and ```subsequences_d1``` may contain different numbers of sequences.  Also, since we need our training set of subsequences (sentences) to be large, we'll add an additional random set of subsequences of each sequence a1 and d1 until we have a total number of ```num_sequences * 2``` training points.  (```num_sequences``` total for each chromosome).
See below:

```python
num_sequences = 10000
a1_temp = []
d1_temp = []

# For a1
for i in range(0,num_sequences-len(subsequences_a1)):
  n1 = np.random.randint(0,len(a1)-sub_len-1)
  a1_temp.append(a1[n1:n1+sub_len])

# For d1
for i in range(0,num_sequences-len(subsequences_d1)):
  n1 = np.random.randint(0,len(d1)-sub_len-1)
  d1_temp.append(d1[n1:n1+sub_len])
```
Then, we'll combine all the subsequences into a single variable called ```corpus_sentences``` containing all the subsequences (sentences).  Note that each subsequence is a single string of length n (ACTGGATCATA...)  

The subsequneces will then be split by their k-mers, giving off the impression of it being a sentence of words. (ACTG GATC ATA...)  The final set of sentences will be assigned to the variable ```corpus_words```.  

```python
corpus_sentences = [];
len_a1 = len(subsequences_a1)
len_d1 = len(subsequences_d1)

for i in range(0,len_a1):
  corpus_sentences.append(subsequences_a1[i])

for i in range(0,len(a1_temp)):
  corpus_sentences.append(a1_temp[i])

for i in range(0,len_d1):
  corpus_sentences.append(subsequences_d1[i])

for i in range(0,len(d1_temp)):
  corpus_sentences.append(d1_temp[i])

corpus_words = []
for i in range(0,len(corpus_sentences)):
  corpus_words_temp = []
  for j in range(0,sub_len-k+1):
    corpus_words_temp.append(corpus_sentences[i][j:j+k])
  corpus_words.append(corpus_words_temp)
```

Next, using an NLP function ```Word2Vec```, we can assign meaningul values to each word in a sentence, transforming them from strings into vectors.  

```python
word2vec_model = Word2Vec(sentences=corpus_words, sg=0, vector_size=100, window=5, min_count=1, negative=5, workers=4, epochs=20)
```

We'll then take all of our sentences, transform them with ```word2vec_model``` for training.

```python
model = Sequential()

# Convolution blocks
for i in range(3):
    model.add(Conv1D(filters=[32, 32, 16][i], kernel_size=[5, 5, 4][i], strides=1, activation='elu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001)
                    ))  # Assuming input sequence length of 100
    model.add(GroupNormalization(groups=[4, 4, 2][i]))
    model.add(MaxPooling1D(pool_size=[4, 4, 2][i], strides=2))
    model.add(Dropout([0.15, 0.2, 0.25][i]))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(32, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                bias_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.004, momentum=0.95),
              loss='binary_crossentropy', metrics=['accuracy'])
```

Lastly, we'll train our network.

```python
num_epochs = 200; batch_sz = 32
model.fit(xtrain_numeric,ytrain,epochs = num_epochs, batch_size = batch_sz, verbose = 1)
```

- $\textbf{Important Note}$:  When ```num_epochs``` $= 200$, the testing accuracy is $\sim 84$%.  It increases as the number of epochs increase.

Once the training is complete, we can test our model with a new G.Hirsutum file.
```python
# Testing on different plant (Train on Cotton 1 -> Test on Cotton 2)
h_test, chromosome_test = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton2.fa')
a1_test = chromosome_test[0]; a1_test = a1_test.upper()
d1_test = chromosome_test[13]; d1_test = d1_test.upper()

corpus_sent_test = []
num_test = 5000

sub_len = 41

# For a1
for i in range(0,num_test):
  n1 = np.random.randint(0,len(a1_test)-sub_len-1)
  corpus_sent_test.append(a1[n1:n1+sub_len])

# For d1
for i in range(0,num_test):
  n1 = np.random.randint(0,len(d1_test)-sub_len-1)
  corpus_sent_test.append(d1[n1:n1+sub_len])

corpus_words_test = []
for i in range(0,len(corpus_sentences)):
  corpus_words_temp = []
  for j in range(0,sub_len-k+1):
    corpus_words_temp.append(corpus_sentences[i][j:j+k])
  corpus_words_test.append(corpus_words_temp)

xtest_kmers = [[word[i:i+k] for i in range(len(word)-k+1)] for sentence in corpus_words_test for word in sentence]
xtest_numeric = np.array([[word2vec_model.wv[word] for word in kmer_list] for kmer_list in xtest_kmers])
xtest_numeric = xtest_numeric.reshape(len(corpus_sent_test), -1, 100)
ytest = np.concatenate([np.zeros(num_test), np.ones(num_test)], axis=0)
```

Test (Evaluate) the model:
```python
model.evaluate(xtest_numeric,ytest)
```

As stated earlier, when the number of epochs is large (above 200), the testing accuracy averages at $84$%.  Based off previous tests, if the number of epochs increases (ideally to 500 or even 1000), then the testing accuracy would get over $90$%.  