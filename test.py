from numpy import load 

with open("data/rule1_ic8932_reflection.npy", 'rb') as spacetime_file:
    s = load(spacetime_file)
    
from metrics.huffman_encoding import HuffmanEncoding

x,table = HuffmanEncoding.encode(image=s)
s_hat = HuffmanEncoding.decode(code=x,image_shape=s.shape,huffman_table=table)
print((s==s_hat).all())