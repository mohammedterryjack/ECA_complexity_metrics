from typing import Optional

from scipy.fft import fft2, ifft2
from numpy import array, count_nonzero, argwhere

from metrics.lossless_fourier_compression_metric.utils import S,R,C,𝝋

class LosslessFourierCompression:
    def __init__(
        self,
        spacetime_evolution:S,
        quantisation_threshold:float=0.5,
        optimal_lossless_filter:Optional[𝝋]=None,
        #TODO:add filter search settings here
    ) -> None:
        self.θ = quantisation_threshold
        self.𝝋_star = self.findNearOptimalFilterFast(spacetime_evolution) if optimal_lossless_filter is None else optimal_lossless_filter
        self.complexity = self.K(spacetime_evolution)

    def C(self,s:S) -> C:
        """lossless compression"""
        self.z = self.F(s)
        return self.𝝋_star(self.z)

    def inverse_C(self, z_hat:C) -> S:
        """lossless decompression"""
        self.r_hat = self.inverse_F(z_hat)
        return self.Q(r=self.r_hat,θ=self.θ)

    def K(self, s:S) -> float:
        """Kolmogorov Complexity"""
        self.s = s
        self.z_hat = self.C(self.s)
        self.s_hat = self.inverse_C(self.z_hat)
        assert not self.ɛ(self.s,self.s_hat), "fourier filter is not lossless!"
        return 1/self.CR(self.z,self.z_hat)

    @staticmethod
    def CR(z:C, z_hat:C) -> float:
        """Compression Ratio"""
        return LosslessFourierCompression.l(z)/LosslessFourierCompression.l(z_hat) 

    @staticmethod
    def ɛ(s:S,s_hat:S) -> float:
        """reconstruction loss"""
        return abs(array(s) - array(s_hat)).sum()

    @staticmethod
    def l(z:C) -> int:
        """compression length"""
        return count_nonzero(z)

    @staticmethod
    def F(r:R) -> C:
        """2D-Fast Fourier Transform"""
        return fft2(r)

    @staticmethod
    def inverse_F(z:C) -> R:
        """inverse 2D-Fast Fourier Transform"""
        return ifft2(z)

    @staticmethod
    def Q(r:R,θ:float) -> S:
        """quantisation step"""
        return array(r>=θ,dtype=int)
    
    @staticmethod
    def findNearOptimalFilterFast(s:S, beam_width:int=1,expansion:int=1,verbose:bool=False) -> 𝝋:
        """biased search for a good lossless filter"""
        ω,T=s.shape
        z = LosslessFourierCompression.F(s)
        nonzero_indexes = argwhere(z.reshape(ω*T))
        indexes_in_magnitude_order = list(filter(
            lambda index:index in nonzero_indexes,
            abs(z).reshape(ω*T).argsort()[::-1]
        ))
        ɸ_best = list(LosslessFourierCompression.ɸ(ω=ω,T=T,l=0))
        loss = ω*T
        losses = [loss]
        compression_length = 1
        while loss>0: 
            ɸ_best,loss = LosslessFourierCompression.leastLossy(
                ɸ_=list(LosslessFourierCompression.expand_compression_length(
                    ɸ_best_smaller=ɸ_best[:beam_width],
                    ω=ω,T=T,
                    next_indexes_flattened=indexes_in_magnitude_order,
                    expansion=expansion
                )),
                z=z
            )
            ɸ_best = list(ɸ_best)
            if verbose:
                print(f"l(𝝋(z))={compression_length}, candidates={len(ɸ_best)}, loss={loss}\n") 
            losses.append(loss)
            compression_length += 1
        #plot(losses)
        #xlabel("Compression Length l(z_hat)")
        #ylabel("Loss ɛ(s,s_hat)")
        #show()
        return list(ɸ_best)[0] 


# def ɸ_optimised(ω:int,T:int, l:int, nonzero_indexes:List[int]) -> Generator[List[List[int]],None,None]:
#     """filters of compression length l"""
#     m_nonzero = zeros(len(nonzero_indexes),dtype=int)
#     m_nonzero[:l] = 1 
#     for m_nonzero_ in multiset_permutations(m_nonzero):
#         nonzero_indexes_ = array(m_nonzero_).nonzero()[0]
#         indexes = nonzero_indexes[nonzero_indexes_]
#         m_ = zeros(ω*T,dtype=int)
#         m_[indexes] = 1
#         m = array(m_).reshape((ω,T))
#         yield 𝝋(mask=m)

# def findOptimalFilter(s:List[List[int]]) -> 𝝋:
#     """greedy search for optimal lossless filter"""
#     ω,T=s.shape
#     size = ω*T
#     z = F(s)
#     total_nonzero_indexes = z.reshape(size).nonzero()[0]
#     for length in range(len(total_nonzero_indexes)+1):
#         for 𝝋_x in ɸ_optimised(ω=ω,T=T,l=length,nonzero_indexes=total_nonzero_indexes):
#             if ɛ(s=s,s_hat=inverse_C(𝝋_x(z)))==0:
#                 print(𝝋_x)
#                 return 𝝋_x


# #============================
# def ɸ(ω:int,T:int, l:int) -> Generator[List[List[int]],None,None]:
#     """filters of compression length l"""
#     m = zeros(ω*T)
#     m[:l] = 1 
#     for m_ in multiset_permutations(m):
#         m_ = array(m_).reshape((ω,T))
#         yield 𝝋(mask=m_)


# def expand_compression_length(
#     ɸ_best_smaller:List[𝝋],
#     next_indexes_flattened:List[int],
#     ω:int,
#     T:int,
#     expansion:int
# ) -> Generator[𝝋,None,None]:
#     for 𝝋_smaller in ɸ_best_smaller:
#         mask_flattened = deepcopy(𝝋_smaller.m.reshape(ω*T))
#         nonzero_indexes = argwhere(mask_flattened)
#         next_indexes_flattened = list(filter(
#             lambda index:index not in nonzero_indexes,
#             next_indexes_flattened
#         ))
#         for 𝝋_expanded in ɸ_expanded_by_index(
#             mask_flattened=mask_flattened,
#             next_indexes_flattened=next_indexes_flattened,
#             ω=ω,T=T,expansion=expansion
#         ):
#             yield 𝝋_expanded

# def leastLossy(ɸ_:List[𝝋], z:List[List[complex]]) -> Tuple[Set[𝝋],int]:
#     losses = list(map(lambda 𝝋_x:ɛ(s=s,s_hat=inverse_C(𝝋_x(z))),ɸ_))
#     min_loss = min(losses)
#     indexes_min_loss = (index for index,loss in enumerate(losses) if loss==min_loss)
#     return set(map(lambda index:ɸ_[index],indexes_min_loss)),min_loss

# def ɸ_expanded_by_index(
#     mask_flattened:List[int],
#     next_indexes_flattened:List[int],
#     ω:int,T:int,expansion:int
# ) -> Generator[𝝋,None,None]:
#     for index in next_indexes_flattened[:expansion]:
#         m = deepcopy(mask_flattened)
#         m[index] = 1
#         yield 𝝋(mask=m.reshape((ω,T)))
