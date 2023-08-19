- In this file, we'll walk through the steps taken to create the convolutional neural network (CNN) which is used to detect the origin of subreads to either A1 or D1 of G.Hirsutum.
- $\textbf{Important Note:}$ The current code is written to handle missing entries as indicated by $N$.  Note that we can use the network as described in ```RNN Missing Values.md``` to fill in the missing entries first, then run it through this pipeline.
- I would suggest creating a conda environment specific for this project so as not to interfere with any dependencies.  

## Section 1:  Importing Packages and Data

We'll start by loading all the necessary packages:
```python
from gensim.models import Word2Vec
import numpy as np
import math
import tensorflow as tf
from itertools import product
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, GroupNormalization, Input
from tensorflow.keras.activations import elu, sigmoid
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
# The cotton file chosen was arbitrarily (you can add any fasta file of interest)
h, seq = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton1.fa')

# In this project, we're considering G.Hirsutum and the relationship between A1 and D1
# Here, we're assigning the chromosomes A1 and D1 accordingly.
# The step .upper() takes all the lowercase letters (a,c,t,g) and capitalizes them.
a1 = seq[0]; a1_len = len(a1)
d1 = seq[13]; d1_len = len(d1)

# If you'd like to look at all the unique entries (bases) of the sequences, use this:
a1_unique = list(set(a1)
d1_unique = list(set(d1))

# Then print
print(a1_unique)
print(d1_unique)

# Some bases may be lowercase, or missing (n or N).  Use this to capitalize all the bases
# otherwise they may be counted as unique entries when we want (A/a, C/c, T/t, G/g, and N/n) to be read the same.
a1 = a1.upper()
d1 = d1.upper()
```

## Section 2:  Generating Subreads for A1/D1 Classification
### Option 1:  Generating Reads of Set Length (not from Negative Binomial)

```python
# Option 1:  Collecting all unique subsequences of length ```read_length``` from A1 and D1 for training.
read_length = 62000

a1_train = []
d1_train = []

a1_train = []
d1_train = []
for i in range(0,a1_len,read_length):
    a1_train.append(a[i:i+read_length])
    if len(a1_train[-1]) < read_length:
        temp = a1_train[-2]
        rem_len = read_length - len(a1_train[-1])
        a1_train[-1] = temp[-rem_len:] + a1_train[-1]  
for i in range(0,d1_len,read_length):
    d1_train.append(d[i:i+read_length])
    if len(d1_train[-1]) < read_length:
        temp = d1_train[-2]
        rem_len = read_length - len(d1_train[-1])
        d1_train[-1] = temp[-rem_len:] + d1_train[-1] 
a1_train_num = len(a1_train)
d1_train_num = len(d1_train)   

temp = [a1_train_num,d1_train_num]
num_train = max(temp)
remainder = abs(a1_train_num-d1_train_num)
min_indx = temp.index(min(temp))
if min_indx == 0:
    for i in range(remainder):
        n1 = np.random.randint(a1_len-read_length-1)
        a1_train.append(a1[n1:n1+read_length])
else:
    for i in range(remainder):
        n1 = np.random.randint(d1_len-read_length-1)
        d1_train.append(d1[n1:n1+read+length])   
```

Then, we'll combine all the subsequences into a single variable called ```corpus_sentences``` containing all the subsequences (sentences).  Note that each subsequence is a single string of length n (ACTGGATCATA...)  

The subsequneces will then be split by their k-mers, giving off the impression of it being a sentence of words. (ACTG GATC ATA...)  The final set of sentences will be assigned to the variable ```corpus_words```.  

```python
k_mers = 11
corpus_sentences = [];
for i in range(num_train):
    corpus_sentences.append(a1_train[i])
for i in range(num_train):
    corpus_sentences.append(d1_train[i])

# Option 1:  Words whose characters overlap (Overlapping k-mers)
# With overlapping k-mers
corpus_words = []
for i in range(0,len(corpus_sentences)):
  corpus_words_temp = []
  for j in range(0,read_length-k_mers+1):
    corpus_words_temp.append(corpus_sentences[i][j:j+k_mers])
  corpus_words.append(corpus_words_temp)

# Option 2:  Words whose characters don't overlap (Non-overlapping k-mers)
# Non overlapping k-mers
corpus_words = []
for string in corpus_sentences:
    sep_sentence = [string[i:i+k_mers] for i in range(0,len(string),k_mers)]
    last_word = sep_sentence[-1]
    if len(last_word) < k_mers:
       prev_segment = sep_sentence[-2]
       num_pad_chars = k - len(last_word)
       last_word = prev_segment[-num_pad_chars:] + last_word
       sep_sentence[-1] = last_segment
    corpus_words.append(sep_sentence)
```

## Section 3:  Sequences $\rightarrow$ Sentences $\rightarrow$ Word Vectors $\rightarrow$ CNN
### 3.1 Training
Next, using an NLP function ```Word2Vec```, we can assign meaningul values to each word in a sentence, transforming them from strings into vectors.  

```python
num_epochs_words = 20
vec_size_words = 100
word2vec_model = Word2Vec(sentences=corpus_words, sg=0, vector_size=vec_size_words, window=5, min_count=5, negative=5, workers=4, epochs=num_epochs_words)
```

We'll then take all of our sentences, transform them with ```word2vec_model``` for training.

```python
# CNN Model
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

```

Once the training is complete, we can test our model with a new G.Hirsutum file.

### 3.2 Testing
```python
# Testing on different plant (Train on Cotton 1 -> Test on Cotton 2)
h_test, chromosome_test = read_fasta_file('/project/90daydata/gbru_sugarcane_seq/Zach/Cotton/cotton2.fa')
a1_test = chromosome_test[0]; a1_test = a1_test.upper()
d1_test = chromosome_test[13]; d1_test = d1_test.upper()

corpus_sent_test = []
num_test = 500

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
