from Vector2 import Vector2
import re
import math

from parse_color import parse_function


class Mat2x3:
    """2D transformation matrix"""

    def __init__(self, mat: "list[list[float]]"):
        self.mat = [
            [float(mat[0][0]), float(mat[0][1]), float(mat[0][2])],
            [float(mat[1][0]), float(mat[1][1]), float(mat[1][2])]
        ]

    def __mul__(a: "Mat2x3", b: "Mat2x3") -> "Mat2x3":
        """Matrix multiplication, returns the matrix when a is applied to b"""
        if type(b) is Vector2:
            return Vector2(
                a.mat[0][0]*b.x + a.mat[0][1]*b.y + a.mat[0][2],
                a.mat[1][0]*b.x + a.mat[1][1]*b.y + a.mat[1][2]
            )
        c = Mat2x3([
            [
                a.mat[0][0]*b.mat[0][0] + a.mat[0][1]*b.mat[1][0],
                a.mat[0][0]*b.mat[0][1] + a.mat[0][1]*b.mat[1][1],
                a.mat[0][0]*b.mat[0][2] + a.mat[0][1]*b.mat[1][2] + a.mat[0][2]
            ],
            [
                a.mat[1][0]*b.mat[0][0] + a.mat[1][1]*b.mat[1][0],
                a.mat[1][0]*b.mat[0][1] + a.mat[1][1]*b.mat[1][1],
                a.mat[1][0]*b.mat[0][2] + a.mat[1][1]*b.mat[1][2] + a.mat[1][2]
            ]
        ])
        return c


def parse_css_transform(transform: str) -> Mat2x3:
    """Construct transformation matrix from a CSS transformation string
       https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
       Not well-tested, may have bug
    """
    transform = transform.strip().replace(",", " ")
    transform = re.sub(r"\s+", " ", transform)
    transform = transform.replace("( ", "(").replace(" )", ")")
    transform = [t.strip()+')' for t in transform.rstrip(')').split(')')]
    matrix = Mat2x3([[1, 0, 0], [0, 1, 0]])
    for ts in transform:
        funname, params = parse_function(ts)
        # matrix
        if funname == "matrix" and len(params) == 6:
            matrix *= Mat2x3([
                [params[0], params[1], params[4]],
                [params[2], params[3], params[5]]
            ])
            continue
        # translate
        if funname == "translate" and len(params) == 2:
            matrix *= Mat2x3([[1, 0, params[0]], [0, 1, params[1]]])
            continue
        if funname in ["translate", "translateX"] and len(params) == 1:
            matrix *= Mat2x3([[1, 0, params[0]], [0, 1, 0]])
            continue
        if funname == "translateY" and len(params) == 1:
            matrix *= Mat2x3([[1, 0, 0], [0, 1, params[0]]])
            continue
        # rotate about a point
        if funname == "rotate" and len(params) == 3:
            a = params[0] * math.pi/180.
            sa, ca = math.sin(a), math.cos(a)
            dx, dy = params[1:3]
            rmat = Mat2x3([[1, 0, dx], [0, 1, dy]]) \
                * Mat2x3([[ca, -sa, 0], [sa, ca, 0]]) \
                * Mat2x3([[1, 0, -dx], [0, 1, -dy]])
            matrix *= rmat
            continue
        # scale
        if funname == "scale" and len(params) == 1:
            matrix *= Mat2x3([[params[0], 0, 0], [0, params[0], 0]])
            continue
        if funname == "scale" and len(params) == 2:
            matrix *= Mat2x3([[params[0], 0, 0], [0, params[1], 0]])
            continue
        # not implemented
        raise ValueError("Transform attribute parsing error", ts)
    return matrix
