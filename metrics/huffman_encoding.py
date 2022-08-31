from typing import Dict, List, Tuple
from numpy import ndarray, array


class HuffmanEncoding:
    @staticmethod
    def compression_ratio(image:ndarray) -> float:
        code,_ = HuffmanEncoding.encode(image)
        return len(''.join(map(str,image.flatten())))/len(code)

    @staticmethod
    def encode(image:ndarray,pixel_stride:int=3) -> Tuple[str,Dict[str,str]]:
        huffman_code_to_original = HuffmanEncoding.huffman_table(image=image)

        encoded = '' 
        x,y = image.shape
        image_flat = image.flatten()

        for index in range(0,x*y,pixel_stride):
            pattern = ''.join(map(str,image_flat[index:index+pixel_stride]))
            encoded += huffman_code_to_original[pattern]

        return encoded, huffman_code_to_original

    @staticmethod
    def decode(code:str, image_shape:Tuple[int,int], huffman_table:Dict[str,str]) -> ndarray:
        table = dict((value,key) for key,value in huffman_table.items())
        result = ''
        while any(code):
            for pattern,decoded in table.items():
                if code.startswith(pattern):
                    result += decoded 
                    code = code[len(pattern):]
                    break
        return array(list(map(int,result))).reshape(image_shape)

    @staticmethod
    def huffman_table(image:ndarray) -> Dict[str,str]:
        return HuffmanEncoding.encoding_table(
            node = HuffmanEncoding.build_tree(
                frequency_table=HuffmanEncoding.frequency_table(
                    image=image
                )
            ) 
        )

    @staticmethod
    def encoding_table(node:dict, encoded='') -> Dict[str,str]:
        if not isinstance(node,dict):
            return {node: encoded}

        table = dict()
        table.update(HuffmanEncoding.encoding_table(
            node=node["left"], 
            encoded=encoded +'0'
        ))
        table.update(HuffmanEncoding.encoding_table(
            node=node["right"], 
            encoded=encoded + '1'
        ))
        return table

    @staticmethod
    def frequency_table(image:ndarray,pixel_stride:int=3) -> List[Tuple[str,int]]:
        table = dict()
        x,y = image.shape
        image_flat = image.flatten()
        for index in range(0,x*y,pixel_stride):
            pixels = ''.join(map(str,image_flat[index:index+pixel_stride]))
            if pixels not in table:
                table[pixels] = 0
            table[pixels] += 1
        return table
 
    @staticmethod
    def build_tree(frequency_table:Dict[str,int]) -> dict:
        nodes = sorted(
            frequency_table.items(), 
            key=lambda pixels_frequency: pixels_frequency[1], 
            reverse=True
        )
        while len(nodes) > 1:
            pixels1, frequency1 = nodes.pop()
            pixels2, frequency2 = nodes.pop()
            nodes.append(
                (
                    dict(left=pixels1,right=pixels2), 
                    frequency1 + frequency2
                )
            )
            nodes.sort(
                key=lambda pixels_frequency: pixels_frequency[1], 
                reverse=True
            )
        return nodes[0][0]