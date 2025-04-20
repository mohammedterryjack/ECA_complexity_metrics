#-----------------------FOR BINARY SEQUENCES----

from scipy.fft import fft, ifft
from numpy import array, count_nonzero, flatnonzero, zeros, ones, inf

class 𝝋:
    def __init__(self, mask:list[int]) -> None:
        self.m = mask
    
    def __hash__(self) -> str:
        return hash(str(self))

    def __eq__(self, other:"𝝋") -> bool:
        return str(self) == str(other)
    
    def __repr__(self) -> str:
        return str(self.m)
    
    def __call__(self, z:list[complex]) -> list[complex]:
        return z*self.m


def Q(r:list[float],θ:float) -> list[int]:
    """quantisation step"""
    return array(r>=θ,dtype=int)

def inverse_C(z_hat:list[complex], θ:float) -> list[int]:
    """lossless decompression"""
    r_hat = ifft(z_hat).real
    return Q(r=r_hat,θ=θ)

def ɛ(s:list[int],s_hat:list[int]) -> float:
    """reconstruction loss"""
    return abs(s - s_hat).sum()


def 𝝋_prime_search(s:list[int], θ:float) -> 𝝋:
    """
    search for near-optimal lossless filter
    by filtering coefficients in order of magnitude
    (from least influential to most)
    """
    losses = list()
    T=len(s)
    z_hat = fft(s)
    m = ones(T)

    for index_larger_coefficient in (
        abs(z_hat.real)+abs(z_hat.imag)        
    ).argsort():
        z_hat[index_larger_coefficient]=0
        loss = ɛ(
            s=s, 
            s_hat=inverse_C(
                z_hat=z_hat, 
                θ=θ
            )
        ) 
        losses.append(loss)
        if not loss:
            m = zeros(T)
            m[flatnonzero(z_hat)]=1

    return 𝝋(mask = m)
    


def C(s:list[int], 𝝋_star:𝝋) -> list[complex]: 
    """lossless compression"""
    z = fft(s)
    return 𝝋_star(z)


def K(s:list[int], 𝝋_star:𝝋, θ:float) -> int:
    """Kolmogorov Complexity"""
    z_hat = C(s=s, 𝝋_star=𝝋_star)
    s_hat = inverse_C(z_hat=z_hat, θ=θ)
    assert not ɛ(s=s,s_hat=s_hat), "fourier filter is not lossless!"
    return count_nonzero(z_hat)


def normalised_fourier_bounded_kolmogorov_complexity_binary_sequences(
    s:list[int]
) -> float:
    θ = 0.5
    𝝋_star = 𝝋_prime_search(
        s=s, 
        θ=θ
    )
    min_description_length = K(
        s=s, 
        𝝋_star=𝝋_star, 
        θ=θ
    )
    max_description_length = len(fft(s))
    return min_description_length/max_description_length


#-----------------------FOR REAL SEQUENCES----


from scipy.fft import fft
def fourier_bounded_kolmogorov_complexity(s:list[float]) -> float:
    z = fft(s)
    z_magnitudes = abs(z.real) + abs(z.imag)
    z_magnitudes_normalised = z_magnitudes/z_magnitudes.max()
    return z_magnitudes_normalised.sum()

#-----------------------



if __name__ == "__main__":
    from numpy import sin, pi
    from random import random

    t = 1000
    s1 = [1 for _ in range(t)]
    s2 = [i%2 for i in range(t)]
    s3 = [i/100 for i in range(t)]
    s4 = [sin(2*pi*0.05*i) for i in range(t)]
    s5 = [random() for _ in range(t)]

    k1 = fourier_bounded_kolmogorov_complexity_real_sequences(s=s1)
    k2 = fourier_bounded_kolmogorov_complexity_real_sequences(s=s2)
    k3 = fourier_bounded_kolmogorov_complexity_real_sequences(s=s3)
    k4 = fourier_bounded_kolmogorov_complexity_real_sequences(s=s4)
    k5 = fourier_bounded_kolmogorov_complexity_real_sequences(s=s5)

    print(k1)
    print(k2)
    print(k3)
    print(k4)
    print(k5)
