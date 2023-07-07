# Input:  FASTA file path
# Output:  Headers,Sequences as lists

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

