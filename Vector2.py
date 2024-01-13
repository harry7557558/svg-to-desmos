# thanks ChatGPT

import math

class Vector2:
    def __init__(self, x_or_vector=0.0, y=None):
        if isinstance(x_or_vector, Vector2):
            self.x = x_or_vector.x
            self.y = x_or_vector.y
        else:
            self.x = float(x_or_vector)
            self.y = float(y) if y is not None else self.x

    def __str__(self):
        return f"Vector2({self.x}, {self.y})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Vector2) and self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        raise TypeError("Unsupported operand type for +: Vector2 and {}".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        raise TypeError("Unsupported operand type for -: Vector2 and {}".format(type(other)))

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector2(self.x * scalar, self.y * scalar)
        raise TypeError("Unsupported operand type for *: Vector2 and {}".format(type(scalar)))

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar != 0:
                return Vector2(self.x / scalar, self.y / scalar)
            raise ValueError("Division by zero is not allowed.")
        raise TypeError("Unsupported operand type for /: Vector2 and {}".format(type(scalar)))

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __pos__(self):
        return Vector2(self.x, self.y)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def length_squared(self):
        return self.x ** 2 + self.y ** 2

    def normalize(self):
        length = self.length()
        if length != 0:
            return self / length
        raise ValueError("Cannot normalize a zero-length vector.")

    def dot(self, other):
        if isinstance(other, Vector2):
            return self.x * other.x + self.y * other.y
        raise TypeError("Unsupported operand type for dot product: Vector2 and {}".format(type(other)))

    def angle_to(self, other):
        if isinstance(other, Vector2):
            dot_product = self.dot(other)
            len_self = self.length()
            len_other = other.length()
            if len_self != 0 and len_other != 0:
                cosine_angle = dot_product / (len_self * len_other)
                return math.acos(max(-1.0, min(1.0, cosine_angle)))
            raise ValueError("Cannot calculate angle with zero-length vectors.")
        raise TypeError("Unsupported operand type for angle calculation: Vector2 and {}".format(type(other)))
