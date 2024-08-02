# Compress a curve represented by a list of points using FFT

from Vector2 import Vector2
import math
import numpy as np
from float_to_str import float_to_str, join_terms


class TrigSpline:
    """Parametric curve defined by trigonometric series"""

    def __init__(self, control_points: "list[Vector2]"):
        """Calculate the coefficients of the spline based on a list of control points
        Additional info:
            Calculated using the Fast Fourier Transform (FFT) algorithm via NumPy.
            If the number of control points is odd,
            the spline is guaranteed to go through all control points.
        Args:
            control_points: a list of control points
        """
        if len(control_points) == 0:
            control_points = [Vector2(0, 0)]
        n_points = len(control_points)
        x_coords = [p.x for p in control_points]
        y_coords = [p.y for p in control_points]
        x_freqs = np.fft.fft(x_coords)[:(n_points+1)//2] * (2.0/n_points)
        x_freqs[0] = np.real(x_freqs[0]) * 0.5
        y_freqs = np.fft.fft(y_coords)[:(n_points+1)//2] * (2.0/n_points)
        y_freqs[0] = np.real(y_freqs[0]) * 0.5
        self._x_cos = np.real(x_freqs)
        self._x_sin = np.imag(x_freqs)
        self._y_cos = np.real(y_freqs)
        self._y_sin = np.imag(y_freqs)

    def evaluate(self, t: float) -> Vector2:
        """Evaluate the curve at a given parameter value"""
        x_cos = self._x_cos * np.cos(2.0*math.pi*np.arange(len(self._x_cos))*t)
        x_sin = self._x_sin * np.sin(2.0*math.pi*np.arange(len(self._x_sin))*t)
        y_cos = self._y_cos * np.cos(2.0*math.pi*np.arange(len(self._y_cos))*t)
        y_sin = self._y_sin * np.sin(2.0*math.pi*np.arange(len(self._y_sin))*t)
        return Vector2(sum(x_cos)+sum(x_sin), sum(y_cos)+sum(y_sin))

    def evaluate_n(self, n: int, raw: bool = False) -> "list[Vector2]":
        """Evaluate the curve at n points with evenly-spaced parameter values
        When raw is True, returns (x, y); When raw is False, returns list[Vector2]"""
        if n < max(len(self._x_cos), len(self._x_sin), len(self._y_cos), len(self._y_sin)):
            # O(MN), should be affordable for up to 2000
            result = []
            for i in range(n):
                result.append(self.evaluate(i/n))
            if raw:
                xs, ys = [], []
                for i in range(n):
                    xs.append(result[i].x)
                    ys.append(result[i].y)
                    return (np.array(xs), np.array(ys))
            return result
        # O(NlogN) using FFT
        x_cos = np.zeros(n)
        x_cos[:len(self._x_cos)] = self._x_cos
        x_sin = np.zeros(n)
        x_sin[:len(self._x_sin)] = self._x_sin
        xs = np.real(np.fft.ifft(x_cos+x_sin*1j)) * n
        y_cos = np.zeros(n)
        y_cos[:len(self._y_cos)] = self._y_cos
        y_sin = np.zeros(n)
        y_sin[:len(self._y_sin)] = self._y_sin
        ys = np.real(np.fft.ifft(y_cos+y_sin*1j)) * n
        if raw:
            return (xs, ys)
        result = []
        for i in range(n):
            result.append(Vector2(xs[i], ys[i]))
        return result

    def get_magnitude(self) -> float:
        """Return a float number that estimates the size of the shape"""
        x_sum = np.sum(self._x_cos[1:]**2) + np.sum(self._x_sin[1:]**2)
        y_sum = np.sum(self._y_cos[1:]**2) + np.sum(self._y_sin[1:]**2)
        return math.sqrt(0.5*(x_sum+y_sum))

    def get_area_approx(self, n: int) -> float:
        """Return the signed area based on coutour integral"""
        x, y = self.evaluate_n(n, True)
        if len(x) < n:
            return 0.0
        idx = np.arange(n)-1
        a = 0.5 * (x[idx] * y - y[idx] * x)
        return a.sum()

    def count_nonzero(self, epsilon: float = 1e-8) -> int:
        """Count the number of sinusoidal basis with non-zero amplitudes
        Additional info:
            Calculates the number of non-zero sinusoidal basis for each dimension
            and choose the maximum one.
        Args:
            epsilon: a number is considered zero if its absolute value is less than this number.
        Returns:
            the number of non-zero sinusoidal basis.
        """
        x_cos_n = np.count_nonzero(abs(self._x_cos) >= epsilon)
        x_sin_n = np.count_nonzero(abs(self._x_sin) >= epsilon)
        y_cos_n = np.count_nonzero(abs(self._y_cos) >= epsilon)
        y_sin_n = np.count_nonzero(abs(self._y_sin) >= epsilon)
        return max(x_cos_n+x_sin_n, y_cos_n+y_sin_n)

    def is_degenerated(self, epsilon: float = 1e-8) -> bool:
        """Test if this curve is degenerated, or, shrinks at one point
        Args:
            epsilon: two values will be considered equal if their difference is less than this
        Returns:
            True if it is degenerated, False if not
        """
        for arr in [self._x_cos, self._x_sin, self._y_cos, self._y_sin]:
            if len(arr) > 0 and np.any(abs(arr[1:]) >= epsilon):
                return True
        return False

    def phase_shift_xs1(self) -> None:
        """Apply a phase shift so the sin(t) term of x is zero
           This sometimes dramatically decreases the number of non-zero terms"""
        if len(self._x_cos) < 2 and len(self._x_sin) < 2:
            return
        # get phase
        xl = max(len(self._x_cos), len(self._x_sin))
        self._x_cos = np.resize(self._x_cos, xl)
        self._x_sin = np.resize(self._x_sin, xl)
        a = math.atan2(self._x_sin[1], self._x_cos[1])
        # shift x
        xi = np.arange(xl)
        ca, sa = np.cos(xi*a), np.sin(xi*a)
        self._x_cos, self._x_sin = (
            self._x_cos*ca+self._x_sin*sa,
            self._x_cos*sa-self._x_sin*ca)
        # shift y
        yl = max(len(self._y_cos), len(self._y_sin))
        yi = np.arange(yl)
        ca, sa = np.cos(yi*a), np.sin(yi*a)
        self._y_cos = np.resize(self._y_cos, yl)
        self._y_sin = np.resize(self._y_sin, yl)
        self._y_cos, self._y_sin = (
            self._y_cos*ca+self._y_sin*sa,
            self._y_cos*sa-self._y_sin*ca)

    def filter_lowest(self, n_waves: int) -> "TrigSpline":
        """Filter frequencies, keep lowest frequencies
        Args:
            n_waves: the number of sinusoidal basis to keep, same for both dimensions
        Returns:
            a trigonometric spline that keeps waves of the lowest n_waves frequencies
        """
        if not n_waves > 0:
            raise ValueError("Number of waves must be positive.")
        result = TrigSpline([Vector2(0, 0)])
        result._x_cos, result._x_sin = self._filter_lowest_dim(
            self._x_cos, self._x_sin, n_waves)
        result._y_cos, result._y_sin = self._filter_lowest_dim(
            self._y_cos, self._y_sin, n_waves)
        return result

    def filter_greatest(self, n_waves: int) -> "TrigSpline":
        """Filter frequencies, keep greatest amplitudes
        Args:
            n_waves: the number of sinusoidal waves to keep, same for both dimensions
        Returns:
            a trigonometric spline that keeps waves of the greatest n_waves amplitudes
        """
        if not n_waves > 0:
            raise ValueError("Number of waves must be positive.")
        result = TrigSpline([Vector2(0, 0)])
        result._x_cos, result._x_sin = self._filter_greatest_dim(
            self._x_cos, self._x_sin, n_waves)
        result._y_cos, result._y_sin = self._filter_greatest_dim(
            self._y_cos, self._y_sin, n_waves)
        return result

    @staticmethod
    def _filter_lowest_dim(a_cos: "list[float]", a_sin: "list[float]", n_waves: int) -> "tuple[list[float], list[float]]":
        """Filter frequencies in one dimension, keep lowest frequencies
        Args:
            a_cos: the amplitudes of cosine waves of frequencies
            a_sin: the amplitudes of sine waves of frequencies, frequency 0 must be zero
            n_waves: the number of sinusoidal basis to keep
        Returns:
            a tuple of filtered cosine and sine waves
        """
        r_cos = [a_cos[0]]
        r_sin = [0.0]
        for i in range(1, n_waves):
            k = (i - 1) // 2 + 1
            if i % 2 == 1:
                r_cos.append(a_cos[k] if k < len(a_cos) else 0.0)
            else:
                r_sin.append(a_sin[k] if k < len(a_sin) else 0.0)
        return (np.array(r_cos), np.array(r_sin))

    @staticmethod
    def _filter_greatest_dim(a_cos: "list[float]", a_sin: "list[float]", n_waves: int) -> "tuple[list[float], list[float]]":
        """Filter frequencies in one dimension, keep greatest amplitudes
        Args:
            a_cos: the amplitudes of cosine waves of frequencies
            a_sin: the amplitudes of sine waves of frequencies
            n_waves: the number of sinusoidal basis to keep
        Returns:
            a tuple of filtered cosine and sine waves
        """
        if n_waves >= len(a_cos) + len(a_sin) - 1:
            return (a_cos[:], a_sin[:])
        # convert waves to a list of tuples of indices and amplitudes
        # non-negative indices for cosine, negative indices for sine
        waves = []
        for k in range(len(a_cos)):
            waves.append((k, a_cos[k]))
        for k in range(1, len(a_sin)):
            waves.append((-k, a_sin[k]))
        # selection sort, terminate when finds n greatest amplitudes
        for j in range(n_waves):
            largest_i = j
            for i in range(j+1, len(waves)):
                if abs(waves[i][1]) > abs(waves[largest_i][1]):
                    largest_i = i
            temp = waves[j]
            waves[j] = waves[largest_i]
            waves[largest_i] = temp
        # convert them back to arrays
        r_cos = []
        r_sin = []
        for i in range(n_waves):
            k, amp = waves[i]
            if k >= 0:
                while len(r_cos) <= k:
                    r_cos.append(0.0)
                r_cos[k] = amp
            else:
                while len(r_sin) <= -k:
                    r_sin.append(0.0)
                r_sin[-k] = amp
        return (np.array(r_cos), np.array(r_sin))

    def to_latex(self, decimals: int) -> str:
        """Get the LateX of the curve
        Returns:
            a string of LaTeX that is compatible with Desmos
        """
        scale = 10.0**decimals
        latex_x, deg_x = TrigSpline._to_latex_dim(
            self._x_cos, self._x_sin, scale)
        latex_y, deg_y = TrigSpline._to_latex_dim(
            self._y_cos, self._y_sin, scale)
        if deg_x or deg_y:
            return ''
        scale = float_to_str(1.0/scale, 12).lstrip('+')
        if scale == "1":
            scale = ""
        return scale+'('+latex_x+','+latex_y+')'

    @staticmethod
    def _to_latex_dim(a_cos: "list[float]", a_sin: "list[float]", scale: float) -> "tuple[str, bool]":
        """Get the LaTeX of the curve to be exported to Desmos, in one dimension
           Returns: (LaTeX, is_degenerate)
        """
        s = []
        degenerate = True
        for k in range(max(len(a_cos), len(a_sin))):
            kt = str(k) + 't'
            a = a_cos[k] if k < len(a_cos) else 0.0
            b = a_sin[k] if k < len(a_sin) else 0.0
            a = float_to_str(scale*a, 0)
            b = float_to_str(scale*b, 0)
            if kt == "0t":
                if a.lstrip('+') != '0':
                    s.append(a)
            else:
                if kt == "1t":
                    kt = "t"
                if a.lstrip('+') != '0':
                    s.append((a, f"c({kt})"))
                    degenerate = False
                if b.lstrip('+') != '0':
                    s.append((b, f"s({kt})"))
                    degenerate = False
        return (join_terms(s), degenerate)
