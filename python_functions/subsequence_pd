# This function is for a specific objective.  It randomly selects a subsequence similar to that of the function "subsequence"
# but it transforms the subsequence in a way that is readable for the RNN (it does so by using pandas, hence the name subsequence_pd)
# Input: sequence (list or numpy array), length_sub (int)
# Output: subsequence of sequence of length "len_sub" (list or numpy array, depending on sequence).


def subsequence(sequence,len_sub):
  n1 = np.random.randint(0,len(sequence)-len_sub-1)
  seq = np.array(seq)
  s_test = seq[n1:n1+length_sub]
  xtemp = s_test[:-1]; ytemp = s_test[1:]
  xdf = pd.DataFrame(xtemp); ydf = pd.DataFrame(ytemp)
  df = pd.DataFrame; df = pd.concat([xdf,ydf], axis=1)
  values = df.values
  X, y = values, values[:, 1]
  X = X.reshape(len(X), 2, 1)
  y = y.reshape(len(y), 1)
  return X, y
