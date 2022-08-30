from zlib import compress, decompress
from numpy import array, ndarray
from typing import Tuple

class ZLIB:
    @staticmethod
    def compress_image(image:ndarray) -> str:
        return compress(
            ''.join(map(str,image.flatten())).encode(),
            level=9
        )
    
    @staticmethod
    def decompress_image(encoded_image:str, image_shape:Tuple[int,int]) -> ndarray:
        return array(list(map(
            int,decompress(encoded_image).decode()
        ))).reshape(image_shape)

    @staticmethod
    def compression_ratio(image:ndarray) -> float:
        x,y = image.shape
        return (x*y)/len(ZLIB.compress_image(image))