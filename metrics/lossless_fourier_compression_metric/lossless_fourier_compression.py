from typing import Optional, List

from scipy.fft import fft2, ifft2
from numpy import array, count_nonzero, flatnonzero, zeros, ones, inf
from matplotlib.pyplot import figure, show

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
        self.complexity = self.K(spacetime_evolution)
        if verbose: self.display()

    def display(self) -> None:
        canvas = figure(figsize=(9,6))

        top_a = canvas.add_subplot(2,5,1)
        top_b = canvas.add_subplot(2,5,2)
        top_c = canvas.add_subplot(2,5,3)
        top_d = canvas.add_subplot(2,5,4)
        top_e = canvas.add_subplot(2,5,5)
        bottom = canvas.add_subplot(2,5,(6,10))

        top_a.set_title('Original\nSpacetime Evolution\ns',fontsize=8)
        top_a.imshow(self.s)
        top_b.set_title('Fourier Transformed\nSpacetime Evolution\n|z|',fontsize=8)
        top_b.imshow(self.z.real)
        top_c.set_title('Filtered Transformation\nSpacetime Evolution\n|z^|',fontsize=8)
        top_c.imshow(self.z_hat.real)
        top_d.set_title('Reconstructed\nSpacetime Evolution\nr^',fontsize=8)
        top_d.imshow(self.r_hat)
        top_e.set_title('Quantised Reconstructed\nSpacetime Evolution\ns^',fontsize=8)
        top_e.imshow(self.s_hat)

        bottom.set_title(f"Kolmogorov Complexity\nK = {self.complexity}",fontsize=9)
        bottom.set_xlabel('Compression Length\nl(z^)')
        bottom.set_ylabel('Reconstruction Loss\nε(s,s^)')
        bottom.plot(self.losses[::-1])
        best_compression_length = self.l(self.z_hat)
        best_loss = self.ɛ(self.s,self.s_hat)
        bottom.plot(best_compression_length, best_loss, 'red',marker=(5, 2))
        bottom.text(best_compression_length,best_loss,best_compression_length,rotation=-20,fontsize=8,horizontalalignment='right')
        bottom.invert_xaxis()

        show()     

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
        m = ones(ω*T)

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
        L = LosslessFourierCompression.l(z_hat)
        return (
            LosslessFourierCompression.l(z)/L 
        ) if L else inf

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
        return ifft2(z).real

    @staticmethod
    def Q(r:R,θ:float) -> S:
        """quantisation step"""
        return array(r>=θ,dtype=int)

