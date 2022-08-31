from lzma import compress, decompress
from numpy import array, ndarray
from typing import Tuple

class LempelZivMarkovChainAlgorithm:
    @staticmethod
    def compress_image(image:ndarray) -> str:
        return compress(
            data=''.join(map(str,image.flatten())).encode(),
        )
    
    @staticmethod
    def decompress_image(encoded_image:str, image_shape:Tuple[int,int]) -> ndarray:
        return array(list(map(
            int,decompress(data=encoded_image).decode()
        ))).reshape(image_shape)

    @staticmethod
    def compression_ratio(image:ndarray) -> float:
        return len(''.join(map(str,image.flatten())).encode())/len(LempelZivMarkovChainAlgorithm.compress_image(image))