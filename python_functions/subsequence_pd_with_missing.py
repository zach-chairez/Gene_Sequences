# This function works similar to "subsequence_pd" except it adds a step of randomly selecting a percentage of the 
# entries to be "missing" (or any other meaningful representation) and labelling them -1.

# Input: sequence (list or numpy array), length_sub (int), perc (float between 0 and 1 to represent a percentage of values to be missing in the subsequence)
# Output: X,y (dataframes used for training the RNN), subseq_original (numpy array containing the original subsequence, note that its length is 1 longer than y).

def subsequence_pd_with_missing(seq,len_sub,perc):
  seq = np.array(seq)
  n1 = np.random.randint(0,len(seq)-len_sub-1)
  subseq_original = seq[n1:n1+len_sub]
  subseq_missing = s_sub.copy() 
  miss_ind = np.random.choice(length_sub,int(length_sub*perc),replace = False)
  s_sub_copy_miss[miss_ind] = -1
  x_temp = s_sub_copy_miss[:-1]; y_temp = s_sub_copy_miss[1:]
  x_df = pd.DataFrame(x_temp); y_df = pd.DataFrame(y_temp)
  df = pd.concat([x_df,y_df], axis = 1)
  values = df.values
  X, y = values, values[:, 1]
  X = X.reshape(len(X), 2, 1)
  y = y.reshape(len(y), 1)
  return X, y, subseq_original
