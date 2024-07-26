# from fasta : fasta length, fasta seq, fasta offset

def get_seqlen_from_fasta(fasta_file):
    with open(fasta_file,'r') as f:
        lines = f.readlines()
    seq = lines[1].strip()
    ids = lines[0]
    offset = ids.split('/')[1].split('_')[0]
    return len(seq), seq, int(offset)
