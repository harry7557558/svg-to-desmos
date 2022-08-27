from distutils.command.clean import clean
import math
from pygame import Vector2
import time
from copy import deepcopy

from sympy import true
from float_to_str import float_to_str, join_terms, join_curves


class Ellipse():

    def __init__(self, cx, cy, rx, ry):
        """Axis-aligned ellipse"""
        self.cx = float(cx)
        self.cy = float(cy)
        self.rx = float(rx)
        self.ry = float(ry)

    def evaluate(self, t: float) -> Vector2:
        """Evaluate the curve at a given parameter value"""
        a = 2.0*math.pi*t
        return Vector2(self.cx + self.rx*math.cos(a),
                       self.cy + self.ry*math.sin(a))

    def evaluate_n(self, n: int) -> list[Vector2]:
        """Evaluate the curve at n points with evenly-spaced parameter values"""
        return [self.evaluate(i/n) for i in range(n)]

    def translate(self, dx, dy) -> None:
        self.cx += dx
        self.cy += dy

    def reverse_y(self) -> None:
        self.cy = -self.cy
        self.ry = -self.ry  # not necessary

    def to_desmos(self, decimals: int) -> str:
        cx = float_to_str(self.cx, decimals)
        cy = float_to_str(self.cy, decimals)
        rx = float_to_str(self.rx, decimals)
        ry = float_to_str(self.ry, decimals)
        x = join_terms([cx, (rx, 'c(t)')])
        y = join_terms([cy, (ry, 's(t)')])
        latex = f"({x},{y})"
        return {
            "latex": latex,
            "parametricDomain": {"max": "1"},
        }


class BezierCurve():
    """Cubic Bézier curve"""

    def __init__(self, control_points: tuple[Vector2]) -> None:
        """Construct a cubic Bézier curve from a list of control points
        Additional info:
            Supports no more than 3 control points.
            https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Specific_cases
        Args:
            control_points: a tuple/list of control points
        """
        def lerp(p, q, t: float):
            return p * (1.0-t) + q * t
        if len(control_points) == 1:
            p = Vector2(control_points[0])
            self._p0 = p
            self._p1 = p
            self._p2 = p
            self._p3 = p
        elif len(control_points) == 2:
            p, q = map(Vector2, control_points)
            self._p0 = p
            self._p1 = lerp(p, q, 1.0/3.0)
            self._p2 = lerp(p, q, 2.0/3.0)
            self._p3 = q
        elif len(control_points) == 3:
            a, b, c = map(Vector2, control_points)
            self._p0 = a
            self._p1 = lerp(a, b, 2.0/3.0)
            self._p2 = lerp(c, b, 2.0/3.0)
            self._p3 = c
        elif len(control_points) == 4:
            a, b, c, d = map(Vector2, control_points)
            self._p0 = a
            self._p1 = b
            self._p2 = c
            self._p3 = d
        else:
            raise ValueError("Unsupported number of control points.")

    def evaluate(self, t: float) -> Vector2:
        """Evaluate the curve at a given parameter value"""
        w0 = (1.0-t)**3
        w1 = 3.0*t*(1.0-t)**2
        w2 = 3.0*t**2*(1.0-t)
        w3 = t**3
        return self._p0*w0 + self._p1*w1 + self._p2*w2 + self._p3*w3

    def evaluate_tangent(self, t: float) -> Vector2:
        """Evaluate the tangent (derivative) of the curve at a given parameter value"""
        w0 = -t*t + 2.0*t - 1.0
        w1 = 3.0*t*t - 4.0*t + 1.0
        w2 = -3.0*t*t + 2.0*t
        w3 = t*t
        return (self._p0*w0 + self._p1*w1 + self._p2*w2 + self._p3*w3) * 3.0

    def evaluate_n(self, n: int) -> list[Vector2]:
        """Evaluate the curve at n points with evenly-spaced parameter values"""
        return [self.evaluate(i/n) for i in range(n)]

    def is_degenerated(self, epsilon: float = 1e-8) -> bool:
        """Test if this curve is degenerated, or, shrinks at one point"""
        return (self._p0 - self._p3).length() < epsilon \
            and (self._p1 - self._p2).length() < epsilon \
            and (self._p0 - self._p1).length() < epsilon

    def translate(self, dx, dy) -> None:
        self._p0 += Vector2(dx, dy)
        self._p1 += Vector2(dx, dy)
        self._p2 += Vector2(dx, dy)
        self._p3 += Vector2(dx, dy)

    def reverse_y(self) -> None:
        self._p0 = Vector2(self._p0.x, -self._p0.y)
        self._p1 = Vector2(self._p1.x, -self._p1.y)
        self._p2 = Vector2(self._p2.x, -self._p2.y)
        self._p3 = Vector2(self._p3.x, -self._p3.y)

    def to_latex(self, decimals: int, scale: float = 1.0) -> str:
        if decimals > 2:
            # compresses quadratics
            p0 = self._p0
            p1 = self._p1*3.0 - self._p0*3.0
            p2 = self._p2*3.0 - self._p1*6.0 + self._p0*3.0
            p3 = self._p3 - self._p2*3.0 + self._p1*3.0 - self._p0
            latex = join_terms([
                float_to_str(scale*p0, decimals),
                (float_to_str(scale*p1, decimals), 'f'),
                (float_to_str(scale*p2, decimals), 'f^2'),
                (float_to_str(scale*p3, decimals), 'f^3')])
        else:
            # requires less decimal places
            p0 = float_to_str(scale*self._p0, decimals)
            p1 = float_to_str(scale*self._p1, decimals)
            p2 = float_to_str(scale*self._p2, decimals)
            p3 = float_to_str(scale*self._p3, decimals)
            p0 = eval(p0.replace('(', 'Vector2('))
            p1 = eval(p1.replace('(', 'Vector2(')) - p0
            p2 = eval(p2.replace('(', 'Vector2(')) - p0
            p3 = eval(p3.replace('(', 'Vector2(')) - p0
            latex = 'b('+','.join([
                float_to_str(p0.x, decimals).lstrip('+'),
                float_to_str(p0.y, decimals).lstrip('+'),
                float_to_str(p1.x, decimals).lstrip('+'),
                float_to_str(p1.y, decimals).lstrip('+'),
                float_to_str(p2.x, decimals).lstrip('+'),
                float_to_str(p2.y, decimals).lstrip('+'),
                float_to_str(p3.x, decimals).lstrip('+'),
                float_to_str(p3.y, decimals).lstrip('+'),
                'f'
            ])+')'
        return latex


class BezierSpline():
    """A continuous list of cubic Bézier curves"""

    def __init__(self) -> None:
        self._curves = []  # curve pieces

    def __len__(self) -> int:
        return len(self._curves)

    def add_curve(self, curve: BezierCurve) -> None:
        """Add a curve to the list of curves"""
        if type(curve) != BezierCurve:
            curve = BezierCurve(curve)
        self._curves.append(deepcopy(curve))

    def translate(self, dx, dy) -> None:
        for curve in self._curves:
            curve.translate(dx, dy)

    def reverse_y(self) -> None:
        for curve in self._curves:
            curve.reverse_y()

    def evaluate(self, t: float) -> Vector2:
        """Evaluate a point at a parameter value.
        Additional info:
            Assign each curve piece an equal parameter interval,
            the point goes down the curves list as t goes from 0 to 1.
        """
        if len(self._curves) == 0:
            raise ValueError("Spline does not contain any curve.")
        it = (len(self._curves) * t) % len(self._curves)
        ti = math.floor(it)
        return self._curves[ti].evaluate(it-ti)

    def evaluate_n(self, n: int, alp: bool = True) -> list[Vector2]:
        if alp:
            return self.evaluate_n_alp(n, False)
        return [self.evaluate(i/n) for i in range(n)]

    def evaluate_n_alp(self, n: int, verbose: bool = True) -> list[Vector2]:
        """Evaluate the curve at n points, attempt to achieve even arc length
            for each arc between two points.
        Additional info:
            The curve must be closed, continuous, and non-degenerate
            in order to achieve the expected result.
        Args:
            n: the number of points in the result, or line segments
            verbose: print time spent on parameterization if True
        Returns:
            A list of n points
        """
        time_start = time.perf_counter()

        # initialize an array of arc lengths
        arc_lengths = []  # arc lengths for each segment
        arc_lengths_psa = [0.0]  # prefix sum array of arc lengths
        for curve in self._curves:
            al = self.n_integrate(
                lambda t: curve.evaluate_tangent(t).length(), 0.0, 1.0)
            arc_lengths.append(al)
            arc_lengths_psa.append(arc_lengths_psa[-1] + al)

        # points
        result = []
        for i in range(n):
            tn = arc_lengths_psa[-1] * (i/n)

            # binary search the curve that tn belongs to
            # find the first index in arc_lengths_psa that is not less than tn
            ti0, ti1 = 0, len(arc_lengths)
            ti = 0
            while ti1 - ti0 > 1:
                ti = (ti0 + ti1) // 2
                if arc_lengths_psa[ti] >= tn:
                    ti1 = ti
                else:
                    ti0 = ti
            if arc_lengths_psa[ti] < tn:
                ti += 1
            ti -= 1

            # get parameters
            tf = tn - arc_lengths_psa[ti]
            t = tf / arc_lengths[ti]

            # sampling
            curve = self._curves[ti]

            def parameter_position_function(tc):
                precise = False
                if precise:  # prevents out of bound
                    l1 = self.n_integrate(
                        lambda t: curve.evaluate_tangent(t).length(), 0.0, tc)
                    l2 = self.n_integrate(
                        lambda t: curve.evaluate_tangent(t).length(), tc, 1.0)
                    return l1 / (l1 + l2) - tf / arc_lengths[ti]
                else:  # faster
                    l1 = self.n_integrate(
                        lambda t: curve.evaluate_tangent(t).length(), 0.0, tc)
                    return l1 - tf

            param_t = self.n_root(parameter_position_function)
            result.append(curve.evaluate(param_t))

        time_end = time.perf_counter()
        if verbose:
            time_elapsed = time_end - time_start
            print(
                "Arc-length parameterization of {} points completed in {:.1f} ms.".format(n, 1000.0*time_elapsed))
        return result

    def is_degenerated(self, epsilon: float = 0.0) -> bool:
        """Test if this curve is degenerated, or, shrinks at one point
        Returns:
            True iff the curve list is empty, False if not
        """
        return len(self._curves) == 0

    def enforce_closed_curve(self, epsilon: float = 0.0):
        new_curves = []
        start_point = None  # start of the current path
        prev_point = None
        for curve in self._curves:
            if prev_point is not None and (curve.evaluate(0) - prev_point).length() > epsilon:
                if start_point is not None and (prev_point - start_point).length() > epsilon:
                    new_curves.append(BezierCurve((prev_point, start_point)))
                start_point = None
            if start_point is None:
                start_point = curve.evaluate(0)
            prev_point = curve.evaluate(1)
            new_curves.append(curve)
        if start_point is not None and (prev_point - start_point).length() > epsilon:
            new_curves.append(BezierCurve((prev_point, start_point)))
        self._curves = new_curves

    def _to_expression(self, decimals: int, factor: bool) -> str:
        pieces = []
        for curve in self._curves:
            if factor:
                latex = curve.to_latex(decimals=0, scale=10**decimals)
            else:
                latex = curve.to_latex(decimals=decimals)
            pieces.append({
                "latex": latex,
                "parametricDomain": {"max": "1"}
            })
        expr = join_curves(pieces)
        if factor and decimals != 0:
            sc = float_to_str(0.1**decimals, 16).lstrip('+')
            expr['latex'] = sc + expr['latex']
        return expr

    def to_desmos(self, decimals: int) -> str:
        spline = BezierSpline()
        for splinet in clean_spline(self):
            spline._curves += splinet._curves
        expr1 = spline._to_expression(decimals, False)
        expr2 = spline._to_expression(decimals, True)
        if len(expr1['latex']) < len(expr2['latex']):
            return expr1
        return expr2

    @staticmethod
    def n_integrate(fun, x0: float, x1: float) -> float:
        """Numerically integrate a function using Gauss-Laguerre quadrature
        Args:
            fun: a one-dimensional function that receives a float and returns a float or a vector
            x0: the lower bound of integral interval
            x1: the higher bound of integral interval
        Returns:
            The numerical integral of the given function in the given interval
        """
        # n_integrate(math.sin, 0.0, 0.5*PI) => 0.999999977197115
        GLN = 4
        GLT = [.06943184420297371238, .33000947820757186759,
               .66999052179242813240, .93056815579702628761]
        GLW = [.17392742256872692868, .32607257743127307131,
               .32607257743127307131, .17392742256872692868]
        s = 0.0
        for i in range(GLN):
            x = x0 + (x1 - x0) * GLT[i]
            s += GLW[i] * fun(x)
        return s * (x1 - x0)

    @staticmethod
    def n_root(fun):
        """Numerically find the root of a function in [0, 1] using secant method.
        Args:
            fun: a function that receives a float and returns a float,
                with a root between 0 and 1.
        Returns:
            The numerical root of the function.
        """
        x0, x1 = 0.0, 1.0
        y0, y1 = fun(x0), fun(x1)
        for i in range(12):
            x2 = x1 - y1 * (x1 - x0) / (y1 - y0)
            y2 = fun(x2)
            x0, y0 = x1, y1
            x1, y1 = x2, y2
            if abs(x1-x0) < 1e-3:
                break
        # print(i, end=' ')  # 3-8 steps
        return x2


def clean_spline(spline: BezierSpline, epsilon: float = 1e-8) -> list[BezierSpline]:
    """Split the spline into a list of continuous splines of non-degenerate points
    Args:
        spline: the Bézier spline to clean
        epsilon: two values are considered equal if their difference is less than this
    Returns:
        a list of "good-quality" Bézier splines.
    """
    splines = []
    if len(spline._curves) == 0:
        return splines
    temp_spline = BezierSpline()
    prev_p = Vector2(float('nan'))
    for curve in spline._curves:
        if curve.is_degenerated(epsilon):
            continue
        if not (curve._p0-prev_p).length() < epsilon:
            if len(temp_spline._curves) != 0:
                splines.append(temp_spline)
            temp_spline = BezierSpline()
        temp_spline.add_curve(curve)
        prev_p = curve._p3
    if len(temp_spline._curves) != 0:
        splines.append(temp_spline)
    return splines
