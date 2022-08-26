from metrics.lossless_fourier_compression_metric import LosslessFourierCompression
from metrics.run_length_encoding import RunLengthEncoding


METRICS = dict(
    #LosslessFourierCompression =lambda s:LosslessFourierCompression(spacetime_evolution=s).complexity,
    Test=lambda s:0.3,
    RunLengthEncoding=lambda s:1/RunLengthEncoding.compression_ratio(image=s),
  )