from metrics.lossless_fourier_compression_metric import LosslessFourierCompression
from metrics.run_length_encoding import RunLengthEncoding
from metrics.huffman_encoding import HuffmanEncoding
from metrics.gzip import GZIP 
from metrics.zlib import ZLIB
from metrics.bz2 import BZ2
from metrics.lzma import LempelZivMarkovChainAlgorithm
from metrics.bdm import BlockDecompositionMethod

bdm = BlockDecompositionMethod()

METRICS = dict(
    LosslessFourierCompression =lambda s:LosslessFourierCompression(spacetime_evolution=s).complexity,
    RunLengthEncoding=lambda s:1/RunLengthEncoding.compression_ratio(image=s),
    HuffmanEncoding=lambda s:1/HuffmanEncoding.compression_ratio(image=s),
    ZLIB=lambda s:1/ZLIB.compression_ratio(image=s),
    GZIP=lambda s:1/GZIP.compression_ratio(image=s),
    BZ2=lambda s:1/BZ2.compression_ratio(image=s),
    LZMA=lambda s:1/LempelZivMarkovChainAlgorithm.compression_ratio(image=s),
    BlockDecompositionMethod=lambda s:bdm.shannon_entropy(image=s)
  )