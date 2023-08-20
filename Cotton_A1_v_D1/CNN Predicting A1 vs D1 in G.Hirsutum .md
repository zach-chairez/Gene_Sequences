- In this file, we'll walk through the steps taken to create the convolutional neural network (CNN) which is used to detect the origin of subreads to either A1 or D1 of G.Hirsutum.
- $\textbf{Important Note:}$ The current code is written to handle missing entries as indicated by $N$.  Note that we can use the network as described in ```RNN Missing Values.md``` to fill in the missing entries first, then run it through this pipeline.

## Section 1:  Importing Packages and Data

We'll start by loading all the necessary packages:
```python
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from itertools import product
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, GroupNormalization, Input
from tensorflow.keras.activations import elu, sigmoid
import scipy
import pandas as pd
from scipy.stats import nbinom
```

The function ```read_fasta_file``` can be found in the folder python_functions.  It's used to load a fasta file path, then outputs the headers and sequences.  We'll first load the following:

```python
def read_fasta_file(file_path):
    headers = []
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_header = None
        current_sequence = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append(current_sequence)
                current_header = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        # Append the last sequence to the lists
        headers.append(current_header)
        sequences.append(current_sequence)
    return headers, sequences

# read_fasta_file reads the file path, then outputs the headers and sequences.
h, seq = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton1.fa')

# Here, we're assigning the chromosomes A1 and D1 accordingly.
# The step .upper() takes all the lowercase letters (a,c,t,g) and capitalizes them.
a1 = seq[0]; a1 = a1.upper()
d1 = seq[13]; d1 = d1.upper()
```

## Section 2:  Generating Subreads for A1/D1 Classification via Negative Binomial Estimation
Next, we'll define two functions (can be found in the folder ```python_functions```:  
- 1. ```find_kmers``` 
- 2.  ```subsequence_all_kmers```

The first function creates a list of all k-mers present in a sequence, while the second function generates subsequences of some length n, ensuring that all k-mers established from the first function are present in the generated subsequences.  This is important as it's necessary to create a "dictionary of words (k-mers)" for our network.  We're going to approach this problem from a natural language processing (NLP) point of view, treating each k-mer as a word in a sequence (or sentence).  

```python

# Find different k-mers in a sequence
def find_kmers(base_string,k):
    unique_permutations = set()
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
        start_index = np.random.randint(0, len(base_string) - n)
        subsequence = base_string[start_index:start_index + n]
        if any(kmer in subsequence for kmer in kmers):
            final_subsequences.append(subsequence)
            kmers -= {kmer for kmer in kmers if kmer in subsequence}
    return final_subsequences

```
Now we'll generate subsequences to create sentences from to build our dictionary of words for our network.  The value of ```read_length``` will be set to a large enough value
so that the negative binomial distribution random values will be less than that (for padding purposes).

Here, we'll import the text file ```CCS_read_length.txt``` to establish parameters for a negative binomial distribution.

```python
# Import text file, assign the frequencies and read_lengths (data_points) accordingly.
df = pd.read_csv('/project/90daydata/gbru_sugarcane_seq/Zach/CCS_read_length.txt', header = None, sep = ' ')
df = df.iloc[:,6:]
frequencies = np.array(df[6]); data_points = np.array(df[7])

# Remove all the nan entries.
nan_indices = np.isnan(data_points)
data_points = data_points[~nan_indices]
frequencies = frequencies[~nan_indices]

# Find sample mean and variance.
sample_mean = np.sum(data_points * frequencies) / np.sum(frequencies)
sample_variance = np.sum((data_points - sample_mean) ** 2 * frequencies) / np.sum(frequencies)

# Compute negative binomial parameters.
n = round((sample_mean ** 2) / (sample_variance - sample_mean))
p = (sample_mean / sample_variance)
```

```python
k_mers = 3
read_length = 2000
subsequences_a1 = subsequence_all_kmers(a1,k_mers,read_length);
subsequences_d1 = subsequence_all_kmers(d1,k_mers,read_length);
```

Note:  The value of ```k_mers``` should be tweaked.  Try values in the interval [2,M], where M is still up in the air.  (I've seen ```k_mers``` as defined as 11).  

The sequences ```subsequences_a1``` and ```subsequences_d1``` may contain different numbers of sequences.  Also, since we need our training set of subsequences (sentences) to be large, we'll add an additional random set of subsequences of each sequence a1 and d1 until we have a total number of ```num_train * 2``` training points.  (```num_train``` total for each chromosome).
See below:

```python
num_train = 100000
a1_temp = []; d1_temp = []
max_a1_start = len(a1) - read_length - 1
max_d1_start = len(d1) - read_length - 1

# For a1
for i in range(0,num_train-len(subsequences_a1)):
  n1 = np.random.randint(0,max_a1_start)
  temp_read = a1[n1:n1+read_length]
  a1_temp.append(temp_read)  

# For d1
for i in range(0,num_train-len(subsequences_d1)):
  n1 = np.random.randint(0,max_d1_start)
  temp_read = d1[n1:n1+read_length]
  d1_temp.append(temp_read)
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
```
Then, take the new padded sentences and create sentences of words.  
```python

# Option 1:  With overlapping k-mers
corpus_words = []
for i in range(0,len(padded_corpus_sentences)):
  corpus_words_temp = []
  for j in range(0,read_length-k_mers+1):
    corpus_words_temp.append(padded_corpus_sentences[i][j:j+k_mers])
  corpus_words.append(corpus_words_temp)

# Option 2:  Non overlapping k-mers
corpus_words = []
read_length = max_length
for string in padded_corpus_sentences:
    sep_sentence = [string[i:i+k_mers] for i in range(0,len(string),k_mers)]
    last_word = sep_sentence[-1]
    if len(last_word) < k_mers:
       prev_segment = sep_sentence[-2]
       num_pad_chars = k_mers - len(last_word)
       last_word = prev_segment[-num_pad_chars:] + last_word
       sep_sentence[-1] = last_word
    corpus_words.append(sep_sentence)
```

## Section 3:  Sequences $\rightarrow$ Sentences $\rightarrow$ Word Vectors $\rightarrow$ CNN
### 3.1 Training
Next, using an NLP function ```Word2Vec```, we can assign meaningul values to each word in a sentence, transforming them from strings into vectors.  

```python
num_epochs_words = 20
vec_size_words = 100
word2vec_model = Word2Vec(sentences=corpus_words, sg=0, vector_size=vec_size_words, window=5, min_count=1, negative=5, workers=4, epochs=num_epochs_words)
```

We'll then take all of our sentences, transform them with ```word2vec_model``` for training.

```python
# CNN Model
# Define the parameters

# Create an Input layer with the desired input shape
input_shape = (read_length - k_mers + 1, vec_size_words)
input_layer = Input(shape=input_shape)

# Convolution blocks
def create_conv_block(x, filters, kernel_size, pool_size, dropout_rate, groups):
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, activation=elu, padding='same', 
               kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = GroupNormalization(groups=groups)(x)
    x = MaxPooling1D(pool_size=pool_size, strides=2)(x)
    x = Dropout(dropout_rate)(x)
    return x

# First Convolution block
conv_block1 = create_conv_block(input_layer, filters=32, kernel_size=5, pool_size=4, dropout_rate=0.15, groups=4)

# Second Convolution block
conv_block2 = create_conv_block(conv_block1, filters=32, kernel_size=5, pool_size=4, dropout_rate=0.2, groups=4)

# Third Convolution block
conv_block3 = create_conv_block(conv_block2, filters=16, kernel_size=4, pool_size=2, dropout_rate=0.25, groups=2)

# Flatten layer
flatten_layer = Flatten()(conv_block3)

# Fully connected layers
fc_layer1 = Dense(32, activation=elu, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))(flatten_layer)
fc_layer2 = Dense(1, activation=sigmoid)(fc_layer1)  # Assuming it's binary classification

# Create the model
model_cnn = tf.keras.models.Model(inputs=input_layer, outputs=fc_layer2)

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.004, momentum=0.95)
model_cnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

```

Lastly, we'll orient our training data for the network, then train.

```python
xtrain_kmers = [[word[i:i+k_mers] for i in range(len(word)-k_mers+1)] for sentence in corpus_words for word in sentence]
xtrain_numeric = np.array([[word2vec_model.wv[word] for word in kmer_list] for kmer_list in xtrain_kmers])
xtrain_numeric = xtrain_numeric.reshape(len(corpus_words), -1, vec_size_words)
ytrain = np.concatenate([np.zeros(num_train), np.ones(num_train)], axis=0)

num_epochs = 100; batch_sz = 32

# CNN Model
model_cnn.fit(xtrain_numeric,ytrain,epochs = num_epochs, batch_size = batch_sz, verbose = 1)
model_cnn.save('/project/90daydata/gbru_sugarcane_seq/Zach_moved/full_cnn_model_neg_binomial.h5')
```

- $\textbf{Important Note}$:  When ```num_epochs``` $= 200$, the testing accuracy is $\sim 84$%.  It increases as the number of epochs increase.

Once the training is complete, we can test our model with a new G.Hirsutum file.

### 3.2 Testing
```python
# Testing on different plant (Train on Cotton 1 -> Test on Cotton 2)
h_test, chromosome_test = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton2.fa')
a1_test = chromosome_test[0]; a1_test = a1_test.upper()
d1_test = chromosome_test[13]; d1_test = d1_test.upper()

corpus_sent_test = []
num_test = 50

# Creating reads from set length
# For a1
for i in range(0,num_test):
  n1 = np.random.randint(0,len(a1_test)-read_length-1)
  corpus_sent_test.append(a1_test[n1:n1+read_length])

# For d1
for i in range(0,num_test):
  n1 = np.random.randint(0,len(d1_test)-read_length-1)
  corpus_sent_test.append(d1_test[n1:n1+read_length])

# With overlapping k-mers
corpus_words_test = []
for i in range(0,len(corpus_sent_test)):
  corpus_words_temp = []
  for j in range(0,read_length-k_mers+1):
    corpus_words_temp.append(corpus_sent_test[i][j:j+k_mers])
  corpus_words_test.append(corpus_words_temp)


# Creating reads via Negative Binomial
# For a1
for i in range(num_test):
  temp_sequence = []; padded_sequence = []
  read_length = nbinom.rvs(n,p)
  n1 = np.random.randint(0,len(a1_test)-read_length-1)
  temp_sequence = a1_test[n1:n1+read_length]
  # padded_sequence = temp_sequence.ljust(read_length, "N")
  # corpus_sent_test.append(padded_sequence)
  corpus_sent_test.append(temp_sequence)

# For d1
for i in range(num_test):
  temp_sequence = []; padded_sequence = []
  read_length = nbinom.rvs(n,p)
  n1 = np.random.randint(0,len(d1_test)-read_length-1)
  temp_sequence = d1_test[n1:n1+read_length]
  # padded_sequence = temp_sequence.ljust(read_length, "N")
  # corpus_sent_test.append(padded_sequence)
  corpus_sent_test.append(temp_sequence)
```
Now that we've generated our subreads to test, we need to go one by one, create random samples of length ```read_length``` as defined earlier, then test each one independently in a voting system. 

```python
num_test_inner = 1000
a1_test_inner = []; d1_test_inner = []
# read_length was defined as 41 in this ReadMe

# For a1
for i in range(num_test_inner):
  n1 = np.random.randint(0,len(a1)-read_length-1)
  a1_temp.append(a1[n1:n1+read_length])

# For d1
for i in range(num_test_inner):
  n1 = np.random.randint(0,len(d1)-read_length-1)
  d1_temp.append(d1[n1:n1+read_length])
```
Then we'll pass each set of inner test sequences to create their own corpus_words variable.

```python
corpus_words_inner_test = []
for i in range(0,len(a1_test_inner)):
  corpus_words_inner_temp = []
  for j in range(0,read_length-k_mers+1):
    corpus_words_inner_temp.append(a1_test_inner[i][j:j+k_mers])
  corpus_words_inner_test.append(corpus_words_inner_temp)
```

```python
xtest_kmers = [[word[i:i+k_mers] for i in range(len(word)-k_mers+1)] for sentence in corpus_words_test for word in sentence]
xtest_numeric = np.array([[word2vec_model.wv[word] for word in kmer_list] for kmer_list in xtest_kmers])
xtest_numeric = xtest_numeric.reshape(len(corpus_sent_test), -1, vec_size_words)
ytest = np.concatenate([np.zeros(num_test), np.ones(num_test)], axis=0)
```

Test (Evaluate) the model:
```python
model.evaluate(xtest_numeric,ytest)
```

As stated earlier, when ```num_epochs``` is large (above 200), the testing accuracy averages at $84$%.  From previous results, accuracy increases as ```num_epochs``` increases.  
