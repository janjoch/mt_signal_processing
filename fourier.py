from multiprocess import Pool

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

import pandas as pd

import interplot as ip


def dft(X):
    N = len(X)
    n = np.arange(N)
    Y = np.empty(N, dtype="complex128")

    for k in range(N):
        Y[k] = np.sum(X * np.exp(-2 * np.pi * 1j * k * n / N))

    return Y


def fft(X):
    N = len(X)
    if N <= 64:
        return dft(X)

    if N % 2:
        return dft(X)

    X1 = X[::2]
    X2 = X[1::2]

    Y1 = fft(X1)
    Y2 = fft(X2)

    n = np.arange(N // 2)
    return np.append(
        Y1 + np.exp(-np.pi * 1j / (N // 2) * n) * Y2,
        Y1 - np.exp(-np.pi * 1j / (N // 2) * n) * Y2,
    )


# Note: this doesn't work recursively, since pool tasks can't have children
def fft_p(X):
    N = len(X)
    if N <= 64:
        return dft(X)

    if N % 2:
        return dft(X)

    X1 = X[::2]
    X2 = X[1::2]

    Y1 = fft_p(X1)
    Y2 = fft_p(X2)

    n = np.arange(N // 2)

    return np.append(
        Y1 + np.exp(-np.pi * 1j / (N // 2) * n) * Y2,
        Y1 - np.exp(-np.pi * 1j / (N // 2) * n) * Y2,
    )


def idft(X):
    N = len(X)
    n = np.arange(N)
    Y = np.empty(N, dtype="complex128")

    for k in range(N):
        Y[k] = np.sum(X * np.exp(2 * np.pi * 1j * k * n / N)) / N

    return Y


class Fourier:

    def __init__(self, t, x, f0=None, comp_mode=None, fourier=fft):
        self.t = t
        self.x = x
        self.N = len(self.x)
        self.fourier = fourier

        self.comp_mode = np.abs if comp_mode is None else comp_mode

        self.dt = t[1] - t[0]
        self.f_N = 1 / self.dt / 2

        self._X = None
        self.f = np.linspace(-self.f_N, self.f_N, self.N, endpoint=False)

        self.f0 = f0
        if f0 is not None:
            self.ppm = self.f / f0 * 1e6

    @property
    def X(self):
        if self._X is None:
            self._X = self.fourier(self.x)

        return self._X

    @property
    def Xp(self):
        return self.X[: self.N // 2]

    @property
    def Xm(self):
        return self.X[self.N // 2 :]

    @property
    def Xmp(self):
        return np.append(self.Xm, self.Xp)

    @staticmethod
    def from_dat(file, *args, **kwargs):
        dat = pd.read_csv(file, delimiter=" ", header=None, names=("t", "x"))
        t = np.array(dat["t"])
        x = np.array(dat["x"])

        return Fourier(t, x, *args, **kwargs)

    def undersample(self, lb_step):
        """
        return a copy of the instance with only the `2**lb_step` data point in the time domain
        """
        return Fourier(
            t=self.t[:: 2**lb_step],
            x=self.x[:: 2**lb_step],
            f0=self.f0,
            comp_mode=self.comp_mode,
        )

    def zerofill(self, lb_factor):
        """
        return a copy of the instance, but with a timeseries `2**lb_factor` longer, data filled with zeroes
        """
        t = np.linspace(
            self.t[0],
            self.t[0] + self.dt * 2**lb_factor * self.N,
            2**lb_factor * self.N,
            endpoint=False,
        )
        x = np.append(self.x, np.zeros(self.N * (2**lb_factor - 1)))
        return Fourier(
            t=t,
            x=x,
            f0=self.f0,
            comp_mode=self.comp_mode,
        )

    def trunc(self, lb_N):
        """
        return a copy of the instance, with only the first `2**lb_N` data
        points
        """
        return Fourier(
            t=self.t[: 2**lb_N],
            x=self.x[: 2**lb_N],
            f0=self.f0,
            comp_mode=self.comp_mode,
        )

    def decay(self, tau):
        def transf(t, x, tau=tau):
            x = x * np.exp(-t / tau)
            return t, x

        return self.transform(transf)

    def lorentz_gauss(self, tau, b):
        def transf(t, x, tau=tau, b=b):
            x = x * np.exp(t / tau - t**2 * b)
            return t, x

        return self.transform(transf)

    def cos(self, tmax):
        def transf(t, x, tmax=tmax):
            h = np.cos(t * np.pi / tmax / 2)
            h[t > tmax] = 0.0
            x = x * h
            return t, x

        return self.transform(transf)

    def sin(self, tmax):
        def transf(t, x, tmax=tmax):
            h = np.sin(t * np.pi / tmax)
            h[t > tmax] = 0.0
            x = x * h
            return t, x

        return self.transform(transf)

    def hanning(self, tmax):
        def transf(t, x, tmax=tmax):
            h = (np.cos(t * np.pi / tmax) + 1) / 2
            h[t > tmax] = 0.0
            x = x * h
            return t, x

        return self.transform(transf)

    def transform(self, transf, *args, **kwargs):
        """
        returns a transformed Fourier instance, defined by `transf(t, x) -> t, x`
        """
        return Fourier(
            *transf(self.t, self.x, *args, **kwargs),
            f0=self.f0,
            comp_mode=self.comp_mode,
        )

    def plot_x(self, fig=None, **kwargs):
        fig = ip.Plot.init(
            fig=fig,
        )
        fig.add_line(self.x, **kwargs)
        return fig

    def plot_X(self, comp_mode=None, fig=None, **kwargs):
        fig = ip.Plot.init(
            fig=fig,
        )
        if comp_mode is None:
            comp_mode = self.comp_mode
        fig.add_line(comp_mode(self.X), **kwargs)
        return fig

    def plot_t(self, fig=None, **kwargs):
        fig = ip.Plot.init(
            fig=fig,
            xlabel="t / s",
            ylabel="amplitude",
        )
        fig.add_line(self.t, self.x, **kwargs)
        return fig

    def plot_f(self, comp_mode=None, slc=slice(None), fig=None, **kwargs):
        fig = ip.Plot.init(
            fig=fig,
            xlabel="f / Hz",
            ylabel="amplitude",
        )
        if comp_mode is None:
            comp_mode = self.comp_mode
        fig.add_line(self.f[slc], comp_mode(self.Xmp)[slc], **kwargs)
        return fig

    def plot_fp(self, *args, **kwargs):
        return self.plot_f(*args, slc=slice(self.N // 2, None), **kwargs)

    def plot_fm(self, *args, **kwargs):
        return self.plot_f(*args, slc=slice(None, self.N // 2), **kwargs)

    def plot_ppm(self, comp_mode=None, slc=slice(None), fig=None, **kwargs):
        fig = ip.Plot.init(
            fig=fig,
            xlabel="sigma / ppm",
            ylabel="amplitude",
        )
        if comp_mode is None:
            comp_mode = self.comp_mode
        fig.add_line(self.ppm[slc], comp_mode(self.Xmp)[slc], **kwargs)
        return fig

    def plot_ppmp(self, *args, **kwargs):
        return self.plot_ppm(*args, slc=slice(self.N // 2, None), **kwargs)
