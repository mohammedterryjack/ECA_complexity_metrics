from typing import Optional, List

from scipy.fft import fft2, ifft2
from numpy import array, count_nonzero, flatnonzero, zeros


from metrics.lossless_fourier_compression_metric.utils import S,R,C,𝝋

class LosslessFourierCompression:
    def __init__(
        self,
        spacetime_evolution:S,
        quantisation_threshold:float=0.5,
        optimal_lossless_filter:Optional[𝝋]=None,
        verbose:bool=False
    ) -> None:
        self.θ = quantisation_threshold
        self.𝝋_star = self.𝝋_prime_search( 
            s=spacetime_evolution,
        ) if optimal_lossless_filter is None else optimal_lossless_filter
        if verbose:
            self.display(
                s=self.s,
                z=self.z,
                z_hat=self.z_hat,
                r_hat=self.r_hat,
                s_hat=self.s_hat,
                losses=self.losses
            )
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

    def 𝝋_prime_search(self, s:S) -> 𝝋:
        """
        search for near-optimal lossless filter
        by filtering coefficients in order of magnitude
        (from least influential to most)
        """
        self.losses = list()
        ω,T=s.shape
        z_hat = self.F(s).reshape(ω*T)
        m = zeros(ω*T)

        for index_larger_coefficient in abs(z_hat.real).argsort():
            z_hat[index_larger_coefficient]=0
            loss = self.ɛ(
                s=s, 
                s_hat=self.inverse_C(z_hat=z_hat.reshape((ω,T)))
            ) 
            self.losses.append(loss)
            if not loss:
                m = zeros(ω*T)
                m[flatnonzero(z_hat)]=1

        return 𝝋(mask = m.reshape((ω,T)))

    @staticmethod
    def CR(z:C, z_hat:C) -> float:
        """Compression Ratio"""
        return LosslessFourierCompression.l(z)/LosslessFourierCompression.l(z_hat) 

    @staticmethod
    def ɛ(s:S,s_hat:S) -> float:
        """reconstruction loss"""
        return abs(s - s_hat).sum()

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
    def display(s:S,z:C,z_hat:C,r_hat:R,s_hat:S,losses:List[float]) -> None:
        #plot(losses)
        #xlabel("Compression Length l(z_hat)")
        #ylabel("Loss ɛ(s,s_hat)")
        #show()
        raise NotImplementedError

