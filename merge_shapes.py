# merge shapes with the same color to reduce bytes

from Vector2 import Vector2
from transform import Mat2x3
from spline import BezierSpline, clean_spline
from trig_spline import TrigSpline
import math
import numpy as np
import svg_to_desmos
import time
from copy import deepcopy

EXPORT_SHAPE_MODE = False

def load_svg_to_trig_splines(filename: str, width: float = None) -> "list[dict]":
    """Load an SVG file to a list of shapes in order
       Each loaded shape contains max/min, fill, trigSplines, desmos"""
    time0 = time.perf_counter()
    shapes, errors = svg_to_desmos.load_svg_shapes(filename)
    time1 = time.perf_counter()
    if len(errors) != 0:
        print(*errors, sep='\n')
    print("SVG parsed in {:.1f}ms".format(1000.0*(time1-time0)))

    # generate polygons
    time0 = time.perf_counter()
    shapes_polygons = []
    for i in range(len(shapes)):
        if (i+1) % 1000 == 0:
            print(f"Polygonising {i+1}/{len(shapes)} shapes...")

        # generate polygons
        curve = shapes[i]['curve']
        polygons = []
        if isinstance(curve, BezierSpline):
            for spline in clean_spline(curve):
                n = max(min(8*len(spline), 256), 32)
                polygons.append(spline.evaluate_n(n))
        else:
            polygons.append(curve.evaluate_n(64))

        # apply transform
        transform = shapes[i]['transform']
        for polygon in polygons:
            for i in range(len(polygon)):
                polygon[i] = transform * polygon[i]

        shapes_polygons.append(polygons)
    time1 = time.perf_counter()
    print("Shapes polygonised in {:.1f}ms".format(1000.0*(time1-time0)))

    # calculate AABB
    minp = Vector2(float('inf'))
    maxp = -minp
    bounds = []
    for polygons in shapes_polygons:
        minp1 = Vector2(float('inf'))
        maxp1 = -minp1
        for polygon in polygons:
            for point in polygon:
                minp1.x = min(minp1.x, point.x)
                minp1.y = min(minp1.y, point.y)
                maxp1.x = max(maxp1.x, point.x)
                maxp1.y = max(maxp1.y, point.y)
        minp.x = min(minp1.x, minp.x)
        minp.y = min(minp1.y, minp.y)
        maxp.x = max(maxp1.x, maxp.x)
        maxp.y = max(maxp1.y, maxp.y)
        bounds.append((minp1, maxp1))
    if width is None:
        scale = 1.0
    else:
        scale = width / (maxp.x - minp.x)
    transform = svg_to_desmos.Mat2x3([
        [scale, 0, 0],
        [0, scale, 0]
    ])

    # FFT/LaTeX
    time0 = time.perf_counter()
    shapes_filtered = []
    for i in range(len(shapes)):
        shape = shapes[i]
        pc = (bounds[i][0]+bounds[i][1])/2
        if not EXPORT_SHAPE_MODE:
            pc *= 0

        # convert to TrigSpline using FFT
        expressions = []
        for polygon in shapes_polygons[i]:
            for _ in range(len(polygon)):
                polygon[_] = transform * (polygon[_]-pc)
            tsp = TrigSpline(polygon)
            mag = tsp.get_magnitude()
            area = abs(tsp.get_area_approx(64))
            if mag < 3.0 or (mag < 20.0 and area < 20.0):
                continue
            tsp.phase_shift_xs1()
            latex = tsp.to_latex(0)
            if latex == "":
                continue
            expressions.append({
                'trigSpline': tsp,
                'latex': latex,
                "parametricDomain": {"max": "1"},
            })
        if len(expressions) == 0:
            continue

        # save TrigSpline and Desmos expression
        shape['trigSplines'] = []
        shape['desmos'] = []
        for expr in expressions:
            shape['trigSplines'].append(expr['trigSpline'])
            shape['desmos'].append({
                'latex': expr['latex'],
                'parametricDomain': expr['parametricDomain']
            })

        # approximate min/max
        shape['min'] = Vector2(float('inf'))
        shape['max'] = -shape['min']
        for tsp in shape['trigSplines']:
            xs, ys = tsp.evaluate_n(256, raw=True)
            shape['min'].x = min(shape['min'].x, np.min(xs))
            shape['min'].y = min(shape['min'].y, np.min(ys))
            shape['max'].x = max(shape['max'].x, np.max(xs))
            shape['max'].y = max(shape['max'].y, np.max(ys))

        shapes_filtered.append(shape)
    shapes = shapes_filtered
    time1 = time.perf_counter()
    print("Polygons processed in {:.1f}ms".format(1000.0*(time1-time0)))

    # debug
    if 0:
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="2400"><g transform="scale(1,-1)">'
        for shape in shapes:
            x = shape['min'].x
            y = shape['min'].y
            width = shape['max'].x - x
            height = shape['max'].y - y
            fill = shape['fill']
            svg += f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="black"/>'
        svg += '</g></svg>'
        open(".svg", 'w').write(svg)

    return shapes


def generate_layering_table(shapes: "list[dict]") -> "list[tuple[set, set]]":
    """Generate a table representing the layering relationship of shapes
    Arg:
        @shapes: a list of shapes returned from `load_svg_to_trig_splines()`
    Returns:
        A list of tuples of two sets
        The first set contains the indices of shapes below the current shape,
        The second set contains the indices of shapes above the current shape
    """
    def is_overlap(shape1, shape2):
        """Test if two shapes (possibly) overlap"""
        if shape1['min'].x > shape2['max'].x or \
                shape2['min'].x > shape1['max'].x:
            return False
        if shape1['min'].y > shape2['max'].y or \
                shape2['min'].y > shape1['max'].y:
            return False
        return True

    # copy shapes so it can be modified
    shapes_1 = []
    for i in range(len(shapes)):
        shape = shapes[i]
        shapes_1.append({
            'min': Vector2(shape['min']),
            'max': Vector2(shape['max']),
            'id': i
        })
    shapes = shapes_1

    # build 2D binary tree
    def build_tree(shapes, root):
        # calculate max/min
        pmin = Vector2(float('infinity'))
        pmax = -pmin
        for shape in shapes:
            pmax.x = max(pmax.x, shape['max'].x)
            pmax.y = max(pmax.y, shape['max'].y)
            pmin.x = min(pmin.x, shape['min'].x)
            pmin.y = min(pmin.y, shape['min'].y)
        root['max'] = pmax
        root['min'] = pmin
        # reach children
        if len(shapes) < 4:
            root['shapes'] = shapes
            return
        # divide x
        if (pmax-pmin).x > (pmax-pmin).y:
            cx = 0.5 * (pmin.x + pmax.x)
            child1, child2 = [], []  # left, right
            for shape in shapes:
                if shape['min'].x < cx:
                    child1.append(shape)
                if shape['max'].x > cx:
                    child2.append(shape)
        # divide y
        else:
            cy = 0.5 * (pmin.y + pmax.y)
            child1, child2 = [], []  # down, up
            for shape in shapes:
                if shape['min'].y < cy:
                    child1.append(shape)
                if shape['max'].y > cy:
                    child2.append(shape)
        # recurse
        if len(child1) in [0, len(shapes)] or len(child2) in [0, len(shapes)]:
            root['shapes'] = shapes
        else:
            root['children'] = [{}, {}]
            build_tree(child1, root['children'][0])
            build_tree(child2, root['children'][1])

    # search in 2D binary tree
    def search_tree(shape, root, overlaps: set):
        # check if the shape overlaps the tree
        if not is_overlap(shape, root):
            return
        # check shape list
        if 'shapes' in root:
            for shape1 in root['shapes']:
                if is_overlap(shape, shape1):
                    overlaps.add(shape1['id'])
        # check children
        if 'children' in root:
            for child in root['children']:
                search_tree(shape, child, overlaps)

    n = len(shapes)
    result = [(set({}), set({})) for i in range(n)]

    # O(n²) bruteforce
    if n < 200:
        for i in range(n):
            for j in range(i):
                if is_overlap(shapes[i], shapes[j]):
                    result[i][0].add(j)
                    result[j][1].add(i)

    # search with a tree, O(nlogn) average case
    else:
        root = {}
        build_tree(shapes, root)
        for i in range(n):
            overlaps = set({})
            search_tree(shapes[i], root, overlaps)
            for j in overlaps:
                if j < i:
                    result[i][0].add(j)
                if j > i:
                    result[i][1].add(j)

    # resolve indirect layering relationship, worst case O(N²)
    for i in range(n):
        belows = result[i][0]
        for j in result[i][0]:
            belows = belows.union(result[j][0])
        result[i] = (belows, result[i][1])
    for i in range(n-1, -1, -1):
        aboves = result[i][1]
        for j in result[i][1]:
            aboves = aboves.union(result[j][1])
        result[i] = (result[i][0], aboves)

    return result


def join_shapes(shapes: "list[dict]") -> "list[dict]":
    """Join a list of shapes with the same color"""
    results = []
    latex_length = 0
    for shape in shapes:
        if latex_length == 0:
            joined = {
                'min': Vector2(float('inf')),
                'max': Vector2(float('-inf')),
                'trigSplines': [],
                'desmos': [],
                'fill': "#000"
            }
        joined['min'].x = min(joined['min'].x, shape['min'].x)
        joined['min'].y = min(joined['min'].y, shape['min'].y)
        joined['max'].x = max(joined['max'].x, shape['max'].x)
        joined['max'].y = max(joined['max'].y, shape['max'].y)
        joined['trigSplines'] += shape['trigSplines']
        joined['desmos'] += shape['desmos']
        joined['fill'] = shape['fill']
        latex_length += sum([len(d['latex']) for d in shape['desmos']])
        # if latex_length >= 10000:
        #     results.append(joined)
        #     latex_length = 0
    if latex_length != 0:
        results.append(joined)
    return results


def collect_shapes_greedy(shapes: "list[dict]",
                          layering_table=None,
                          effective_shapes=None) -> "list[dict]":
    """Merge shapes using greedy algorithm, pull out the most occuring color
       Average case O(NlogN), worst case O(N²logN) ??
    """
    if layering_table is None:
        layering_table = generate_layering_table(shapes)
    if effective_shapes is None:
        effective_shapes = set(range(len(shapes)))
    if len(effective_shapes) == 0:
        return []

    # get colors
    colors = {}
    for i in effective_shapes:
        fill = shapes[i]['fill']
        if fill not in colors:
            colors[fill] = set({})
        colors[fill].add(i)
    assert len(colors) > 0
    if len(colors) == 1:
        return join_shapes([shapes[i] for i in effective_shapes])

    # select the most-occuring color
    mcolor = ""
    for color in colors:
        if mcolor == "" or len(colors[color]) > len(colors[mcolor]):
            mcolor = color

    # pull out disjoint elements
    c_indices = set(colors[mcolor])
    m_indices = set({})
    for i in colors[mcolor]:
        non_overlap = True
        for j in list(layering_table[i][0]) + list(layering_table[i][1]):
            if j in m_indices:
                i0, i1 = min(i, j), max(i, j)
                for k in range(i0, i1+1):
                    if k not in c_indices:
                        non_overlap = False
                        break
            if not non_overlap:
                break
        if non_overlap:
            m_indices.add(i)

    # join elements
    m_shapes = []
    for i in m_indices:
        m_shapes.append(shapes[i])
        # clean-up
        effective_shapes.remove(i)
        colors[mcolor].remove(i)
    joined = join_shapes(m_shapes)
    if len(colors[mcolor]) == 0:
        colors.pop(mcolor)

    # split into bottom and top halfs
    ushapes = set({})  # to upper branch
    lshapes = set({})  # to lower branch, default
    for i in m_indices:
        for j in layering_table[i][1]:
            if j not in effective_shapes:
                continue
            ushapes.add(j)
        for j in layering_table[i][0]:
            assert j not in ushapes
    for i in effective_shapes:
        if i not in ushapes:
            lshapes.add(i)
    # print(len(ushapes), len(lshapes))

    # recursively collect upper and lower shapes
    ucollect = collect_shapes_greedy(shapes, layering_table, ushapes)
    lcollect = collect_shapes_greedy(shapes, layering_table, lshapes)
    return lcollect + joined + ucollect


def split_large_shapes(shapes: "list[dict]") -> "list[dict]":
    """Prevent "Definitions are nested too deeply" error"""
    new_shapes = []
    for shape in shapes:
        expr_list = []
        len_count = 0
        for expr in shape['desmos']:
            expr_list.append(expr)
            len_count += len(expr['latex'])
            if len_count > 2000:
                new_shapes.append({
                    'fill': shape['fill'],
                    'desmos': expr_list
                })
                expr_list = []
                len_count = 0
        if len_count > 0:
            new_shapes.append({
                'fill': shape['fill'],
                'desmos': expr_list
            })
    return new_shapes


def get_latex_translation(latex: str) -> "tuple[tuple[str, str], str]":
    """Get the translation of the LaTeX expression of an (x,y) trigonometric series,
        as well as the expression with translation removed"""
    def rip_latex(latex, i0) -> "tuple[str, str]":
        val = ""
        i = i0
        if latex[i] in ['+', '-']:
            val += latex[i]
            i += 1
        while latex[i] not in ['+', '-']:
            if latex[i] not in "0123456789.":
                val = "0"
                i = i0
                break
            val += latex[i]
            i += 1
        if latex[i] == '+':
            i += 1
        return (val, latex[:i0]+latex[i:])
    dx, latex = rip_latex(latex, latex.find('(')+1)
    dy, latex = rip_latex(latex, latex.find(',')+1)
    return (dx, dy, latex)


def extract_common_latex(shapes: "list[dict]") -> "list[dict]":
    """Extract LaTeX expressions that are translations and define them in an expression
    Args:
        shapes: a list of shapes, usually returned by `collect_shapes_greedy()`
    Returns:
        A list of Desmos definitions of functions
        shapes will be automatically compressed"""

    # store the approximate number of saved bytes when applied
    latexes = {}
    for shape in shapes:
        for expr in shape['desmos']:
            dx, dy, latex = get_latex_translation(expr['latex'])
            saved_bytes = len(latex)-10
            if latex not in latexes:
                latexes[latex] = -(len(latex)+90)
            latexes[latex] += saved_bytes

    # apply
    expr_id = 0
    comp_dict = {}
    expressions = []
    for shape in shapes:
        for expr in shape['desmos']:
            dx, dy, latex = get_latex_translation(expr['latex'])
            if not latexes[latex] > 0:
                continue
            if latex not in comp_dict:
                comp_dict[latex] = "a_{" + str(expr_id) + "}"
                expressions.append({
                    "type": "expression",
                    "id": f"a{expr_id}",
                    "color": "#000",
                    "latex": f"{comp_dict[latex]}(t)={latex}",
                    "hidden": True
                })
                expr_id += 1
            expr['latex'] = f"({dx},{dy})+{comp_dict[latex]}(t)"

    return expressions
