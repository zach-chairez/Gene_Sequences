Goal:  To predict missing values from a gene sequence via a RNN.
What we need:
 1.  FASTA files of interest:  For instance, in ```/project/90daydata/gbru_sugarcane_seq/Zach/Cotton```, you'll see a list of files labeled
     ```Cotton1.fa, Cotton2.fa,...,Cotton8.fa```.  These are eight (8) different G.Hirsutum genomes pulled from CottonGen.
     
 2.  For training/testing, we can train on ```Cotton1,2,..7.fa``` and test on ```Cotton8.fa``` (or any combination really, as long as we leave one for testing)
   
 3.  The libraries needed for this project are: ```numpy, pandas, tensorflow```
   
 4.  We can also train on the sugarcane genomes (right now I have the R570 and Colombian which are located in ```/project/90daydata/gbru_sugarcane_seq/Zach/Sugarcane```
     and labeled accordinly.  Note that the files that end in ```"_10"``` are the isolated assemblies with the first 10 chromosomes and headers.  

#### Import necessary libraries
```python

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
```

#### I could leave a note here and say that I can make a loop 
#### to iterate over an entire folder of fasta files of interest.
```python
# PUT THE FUNCTION HERE!
```

#### Read in the fasta file of interest (using read_fasta_file from the folder "python_functions")
```python
path_to_fasta_file = '/path/to/fasta_file/here.fasta'
headers,sequences = read_fasta_file(path_to_fasta_file)
```

#### Map the bases to integers for each sequence in sequences (using change_bases_to_int from the folder "python_functions")
```python
mapped_sequences = change_bases_to_int(sequences)
```

#### This next part assume we're training a RNN to learn about a single chromosome.
#### If we wanted to look at cotton:     $A1 \rightarrow 0, A2 \rightarrow 1, ... , A13 \rightarrow 12, D1 \rightarrow 13, D2 \rightarrow 14, ..., D13 \rightarrow 25$
#### If we wanted to look at sugarcane:  $C1 \rightarrow 0, C2 \rightarrow 1, ..., C10 \rightarrow 9$
#### The j^th chromosome is the $(j-1)^{th}$ index in ```mapped_sequences```.
```python
chromosome_j = mapped_sequences[j-1]
```

#### Create and compile the RNN
```python
model = Sequential()
model.add(Masking(mask_value =-1, input_shape = (2,1)))
model.add(LSTM(5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

#### Train the RNN on (n) random subsequences of the sequence of interest with length "len_sub"
#### You can set... 
##### ```verbose = 0``` (no output while its running), 
##### ```verbose = 1``` (shows progress bars and metrics during each batch)
##### ```verbose = 2``` (shows metrics at the end of each epoch)
```python
n = 100; len_sub = 10000; 
for i in range(n):
 X, y, _ = subsequence_pd(chromosome_j,len_sub)
 model.fit(X, y, epochs=1, batch_size=1, verbose=1)
```

#### Test the RNN on a random subsequence of length "len_sub"
#### Note that len_sub here doesn't have to be the same length as len_sub above, you can fill in missing entries
#### for a subsequence of any length (assuming its less than the length of the full original sequence).
#### Ideally, what we do here is take another chromosome_j from a different cotton (or sugarcane) plant to test
#### For the test plant of interest, reperform ```path_to_fasta_file``` to ```chromosome_j``` to get a new ```chromosome_j```, call it ```chromosome_j_test```

#### Test and check accuracy of predictions.
```python
len_sub = 10000; perc_miss = 0.05;
x_test,y_test,subseq = subsequence_pd_with_missing(chromosome_j_test,len_sub,perc_miss)

subseq_original = subseq.copy()
subseq_original = subseq_original.reshape((len(subseq_original),1))
y_test_original = y_test.copy()
miss_y = np.where(y_test == -1)[0]
miss_y.sort()

while len(miss_y) != 0:
  y_predicted = model.predict(x_test)
  subseq[miss_y[0]+1] = np.round(y_predicted[miss_y[0]])
  x_temp = subseq[:-1]; y_temp = subseq[1:]
  x_df = pd.DataFrame(x_temp); y_df = pd.DataFrame(y_temp)
  df = pd.concat([x_df,y_df], axis = 1)
  values = df.values
  x_test, y_test = values, values[:, 1]
  x_test = x_test.reshape(len(x_test), 2, 1)
  y_test = y_test.reshape(len(y_test),1)
  miss_y = miss_y[1:]

miss = np.where(y_test_original == -1)[0]; len_miss = len(miss)
hit_or_miss = abs(np.round(y_predicted[miss])) == subseq_original[miss+1]
accuracy = sum(sum(hit_or_miss))/len_miss*100
```
The testing accuracy has been steadily in the upper 90's (usually 98 - 100%).  We've done different combinations of training/testing files and the results are similar.   

