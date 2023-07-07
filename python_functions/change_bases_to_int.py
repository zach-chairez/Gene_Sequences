# Input:  The sequences that were outputted by the function read_fasta_file
# Output:  The same sequences but now mapped to integers

def change_bases_to_int(sequence):
  seq_new = []; gene_mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3,'a':0,'c':1,'t':2,'g':3}
  for seq in sequence:
    seq_new.append([gene_mapping[base] if base in gene_mapping else -1 for base in seq])
  return seq_new
