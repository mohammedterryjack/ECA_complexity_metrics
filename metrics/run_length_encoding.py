from typing import Generator, Tuple, List
from numpy import ndarray, array

class RunLengthEncoding:
    @staticmethod
    def compression_ratio(image:ndarray) -> float:
        x,y = image.shape
        return (x*y)/len(list(RunLengthEncoding.encode(image)))

    @staticmethod
    def encode(image:ndarray) -> Generator[Tuple[int,int],None,None]:
        maximum_run_length = 2**8
        count = 0
        previous_pixel = None
        for pixel in image.flatten():
            if previous_pixel is None:
                previous_pixel=pixel
                count += 1
                continue
            if previous_pixel!=pixel:
                yield previous_pixel,count
                previous_pixel=pixel
                count=1
                continue
            if count<maximum_run_length:
                count+=1
                continue
            yield previous_pixel,count 
            previous_pixel=pixel
            count=1
        yield previous_pixel,count 

    @staticmethod
    def decode(encoded:List[Tuple[int,int]], shape:Tuple[int,int]) -> ndarray:
        decoded=list()
        for pattern,count in encoded:
            decoded.extend([pattern]*count)
        return array(decoded).reshape(shape)