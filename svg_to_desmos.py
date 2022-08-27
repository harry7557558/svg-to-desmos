from pygame import Vector2
from spline import Ellipse, BezierCurve, BezierSpline, clean_spline
from trig_spline import TrigSpline
from float_to_str import join_curves
from transform import Mat2x3, parse_css_transform
from parse_color import parse_color
import merge_shapes
from copy import deepcopy
import math
import numpy as np
import xml.dom.minidom
import re
import json


def parse_path(s: str) -> BezierSpline:
    """Parse an SVG path
    Additional info:
        Translated from a JavaScript function I wrote previously.
        The specification of SVG path string can be found at
        https://www.w3.org/TR/SVG/paths.html#DProperty
    Args:
        s: the SVG path string
    Returns:
        a parsed spline, or None if there is an error during parsing
    """
    s = s.strip()
    if len(s) == 0:
        return BezierSpline()

    cmd = ''  # MZLHVCSQTA
    p_start = Vector2(0, 0)  # starting point
    p0 = Vector2(0, 0)  # ending point
    s_prev = Vector2(0, 0)
    t_prev = Vector2(0, 0)  # reflection in smoothed curves
    spline = BezierSpline()  # result
    cont_count = 0  # number of points after a letter

    d = 0  # index
    while d < len(s):

        # parsing functions

        def is_float(c: str) -> bool:
            if c in ['-', '.']:
                return True
            try:
                x = float(c)
            except ValueError:
                return False
            return not math.isnan(x)

        def read_float() -> float:
            """Read the next float in s and increase d"""
            nonlocal s, d
            while d < len(s) and re.match(r"^[\s\,]$", s[d]):
                d += 1
            if d >= len(s) or not is_float(s[d]):
                return float('nan')
            ns = ""
            while d < len(s):
                if s[d] in "0123456789" \
                    or (s[d] in '+-' and (ns == "" or ns[-1] == 'e')) \
                        or (s[d] in "Ee." and s[d].lower() not in ns):
                    ns += s[d].lower()
                else:
                    break
                d += 1
            try:
                return float(ns)
            except ValueError:
                return float('nan')

        def read_point() -> Vector2:
            """Read the next point in s and increase d"""
            x = read_float()
            y = read_float()
            return Vector2(x, y)

        # get command
        while d < len(s) and re.match(r"^[\s\,]$", s[d]):
            d += 1
        if s[d].upper() in "MZLHVCSQT":
            cmd = s[d]
            d += 1
            cont_count = 0
        elif not is_float(s[d]):
            print("SVG path parsing error: float")
            break

        # case work for each command

        if cmd in "Mm":  # move
            p = read_point()
            if cmd.islower():
                p += p0
            p_start = p
            if cont_count > 0:
                spline.add_curve(BezierCurve((p0, p_start)))
            p0 = p

        elif cmd in "Zz":  # close
            spline.add_curve(BezierCurve((p0, p_start)))
            p0 = p_start

        elif cmd in "Ll":  # line
            p = read_point()
            if cmd.islower():
                p += p0
            spline.add_curve(BezierCurve((p0, p)))
            p0 = p

        elif cmd in "Hh":  # horizontal line
            x = read_float()
            if cmd.islower():
                p = Vector2(p0.x+x, p0.y)
            else:
                p = Vector2(x, p0.y)
            spline.add_curve(BezierCurve((p0, p)))
            p0 = p

        elif cmd in "Vv":  # vertical line
            y = read_float()
            if cmd.islower():
                p = Vector2(p0.x, p0.y+y)
            else:
                p = Vector2(p0.x, y)
            spline.add_curve(BezierCurve((p0, p)))
            p0 = p

        elif cmd in "Cc":  # cubic bezier curve
            p1 = read_point()
            p2 = read_point()
            p3 = read_point()
            if cmd.islower():
                p1 += p0
                p2 += p0
                p3 += p0
            spline.add_curve(BezierCurve((p0, p1, p2, p3)))
            s_prev = p2
            p0 = p3

        elif cmd in "Ss":  # smooth cubic bezier curve
            p2 = read_point()
            p3 = read_point()
            if cmd.islower():
                p2 += p0
                p3 += p0
            p1 = 2.0*p0 - s_prev
            spline.add_curve(BezierCurve((p0, p1, p2, p3)))
            s_prev = p2
            p0 = p3

        elif cmd in "Qq":  # quadratic bezier curve
            p1 = read_point()
            p2 = read_point()
            if cmd.islower():
                p1 += p0
                p2 += p0
            spline.add_curve(BezierCurve((p0, p1, p2)))
            t_prev = p1
            p0 = p2

        elif cmd in "Tt":  # smooth quadratic bezier curve
            p2 = read_point()
            if cmd.islower():
                p2 += p0
            p1 = 2.0*p0 - t_prev
            spline.add_curve(BezierCurve((p0, p1, p2)))
            t_prev = p1
            p0 = p2

        else:  # elliptic arc ?!
            raise ValueError("SVG path parsing error: elliptic arc")

        cont_count += 1

        if cmd not in "CcSs":
            s_prev = p0
        if cmd not in "QqTt":
            t_prev = p0

    spline.enforce_closed_curve(1e-6)
    return spline


def load_svg_shapes(filename: str):
    doc = xml.dom.minidom.parse(filename)
    svg = None
    for child in doc.childNodes:
        if child.nodeName == "svg":
            svg = child
            # break
    if svg is None:
        raise ValueError("No SVG tag detected.")

    unsupported_node_names = set({})

    def parse_node(node, existing_attributes={}, defs={}):
        nodeName = node.nodeName
        attributes = deepcopy(existing_attributes)
        new_attributes = {}
        if hasattr(node.attributes, 'items'):
            for (attrib, value) in dict(node.attributes.items()).items():
                if attrib == 'transform' and attrib in attributes:
                    value = attributes[attrib] + value
                attributes[attrib] = value
                new_attributes[attrib] = value

        transform = Mat2x3([[1, 0, 0], [0, -1, 0]])
        if 'transform' in attributes:
            transform *= parse_css_transform(attributes['transform'])

        if 'style' in attributes:
            styles = [s.strip() for s in attributes['style'].split(';')]
            for style in styles:
                if re.match(r"^fill\s*\:", style):
                    attributes['fill'] = style[style.find(':')+1:]
        if 'fill' not in attributes or attributes['fill'] in ["currentColor"]:
            attributes['fill'] = '#000'
        attributes['fill'] = parse_color(attributes['fill'])
        if attributes['fill'] is None:
            return []

        if nodeName in ["#text"]:
            return []

        if nodeName in ["g", "switch"]:
            shapes = []
            for child in node.childNodes:
                shapes += parse_node(child, attributes, defs)
            return shapes

        if nodeName == "defs":
            new_defs = []
            for child in node.childNodes:
                new_defs += parse_node(child, attributes, defs)
            for new_def in new_defs:
                attributes = new_def['attributes']
                new_def['attributes'] = {}
                if 'id' in attributes:
                    defs['#'+attributes['id']] = new_def
                if 'class' in attributes:
                    defs['.'+attributes['class']] = new_def
            return []

        if nodeName == "use":
            href = attributes['href'] if 'href' in attributes else attributes['xlink:href']
            if href not in defs:
                return []
            shape = defs[href]
            assert 'x' not in attributes and 'y' not in attributes and \
                'width' not in attributes and 'height' not in attributes
            return [{
                'fill': attributes['fill'],
                'curve': shape['curve'],
                'transform': transform,
                'attributes': new_attributes
            }]

        if nodeName == "path":
            spline = parse_path(attributes['d'])
            if spline is None:
                raise ValueError("SVG path parsing error.")
            shape = spline

        elif nodeName in ["polygon", "polyline"]:
            if 'points' not in attributes:
                return []
            spline = parse_path('M'+attributes['points'])
            if spline is None:
                raise ValueError("SVG path parsing error.")
            shape = spline

        elif nodeName == "rect":
            x = float(attributes['x'])
            y = float(attributes['y'])
            w = float(attributes['width'])
            h = float(attributes['height'])
            spline = parse_path(f"M{x},{y}h{w}v{h}h{-w}z")
            assert len(spline) == 4
            shape = spline

        elif nodeName == "ellipse":
            ellipse = Ellipse(attributes['cx'], attributes['cy'],
                              attributes['rx'], attributes['ry'])
            shape = ellipse

        elif nodeName == "circle":
            ellipse = Ellipse(attributes['cx'], attributes['cy'],
                              attributes['r'], attributes['r'])
            shape = ellipse

        else:
            if nodeName not in unsupported_node_names:
                print("Unsupported node name:", nodeName)
                unsupported_node_names.add(nodeName)
            return []

        return [{
            'fill': attributes['fill'],
            'curve': shape,
            'transform': transform,
            'attributes': new_attributes
        }]

    shapes = []
    errors = []
    for child in svg.childNodes:
        try:
            shapes += parse_node(child)
        except BaseException as err:
            raise err
            errors.append(err)
    return (shapes, errors)


def shapes_to_desmos(shapes: list[dict], expressions_app: list[dict] = []) -> dict:
    """Fit a list of shapes to Desmos using FFT compression
       @shapes: returned from `merge_shapes.collect_shapes_greedy()`
       @expressions_app: append to the list of expressions
    """

    expressions_list = [
        {
            "type": "expression",
            "id": "c",
            "color": "#000",
            "latex": "c(t)=\\cos(2t\\pi)",
            "hidden": True
        },
        {
            "type": "expression",
            "id": "s",
            "color": "#000",
            "latex": "s(t)=\\sin(2t\\pi)",
            "hidden": True
        },
    ]
    expressions_list += expressions_app

    for i in range(len(shapes)):
        shape = shapes[i]
        expression = join_curves(shape['desmos'])

        expression['parametricDomain']['min'] = ''
        expression['domain'] = {
            # deprecated ?! still adding up bytes
            'min': '0',
            'max': expression['parametricDomain']['max']
        }
        expression['color'] = shape['fill']
        expression['fill'] = True
        expression['lines'] = False  # more bytes but much faster
        expression['fillOpacity'] = '1'
        expression['type'] = "expression"
        expression['id'] = str(i+1)
        expressions_list.append(expression)
    return expressions_list


if __name__ == "__main__":

    def one_svg_to_desmos_merge(filepath, scale: float):
        shapes = merge_shapes.load_svg_to_trig_splines(
            filepath, scale)
        print(len(shapes), "shapes loaded.")
        shapes = merge_shapes.collect_shapes_greedy(shapes)
        shapes = merge_shapes.split_large_shapes(shapes)
        print("Merged:", len(shapes), "expressions.")
        commons = []
        commons = merge_shapes.extract_common_latex(shapes)
        print(len(commons), "common expressions extracted.")
        expressions = shapes_to_desmos(shapes, commons)
        expressions = json.dumps(expressions, separators=(',', ':'))
        expressions = f"var s=Calc.getState();s['expressions']['list']={expressions};Calc.setState(s);"
        # print(expressions)
        open(".desmos", 'w').write(expressions)
        print(len(expressions), "chars")

    # filename, width = "test-svg/uoft-logo.svg", 2000
    # filename, width = "test-svg/es2t6.svg", 6000
    # filename, width = "test-svg/World_map_blank_without_borders.svg", 1200
    # filename, width = "test-svg/equation-2.svg", 8000
    # filename, width = "test-svg/equation-3.svg", 6000
    # filename, width = "test-svg/hermit_crab.svg", 2000
    # filename, width = "test-svg/chinese_paper_cutting.svg", 4000
    filename, width = "test-svg/Python3-powered_hello-world.svg", 2000
    # filename, width = "test-svg/Frog_(2546)_-_The_Noun_Project.svg", 2000
    one_svg_to_desmos_merge(filename, width)
