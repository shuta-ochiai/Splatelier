#!/usr/bin/env python3
"""Writes the furniture STEP fixtures used by the STEP import tests.

The files are AP214 B-reps in millimetres, Z up, as a CAD tool would export them: boxes made of
planar faces and straight edges, cylinders with a seam edge and circular rims, a colour on every
solid, and one assembly whose parts are placed by transforms (the bookshelf reuses one shelf part
five times).

    python3 scripts/make_step_fixtures.py [output directory]
"""

import math
import pathlib
import sys

DEFAULT_OUT = pathlib.Path(__file__).resolve().parent.parent / "Tests/SplatViewerKitTests/Fixtures/STEP"


def real(v):
    """A STEP REAL: always has a decimal point (`5.`, `0.25`, `1.E-07`)."""
    if v == 0:
        return "0."
    s = repr(float(v))
    if "e" in s or "E" in s:
        mantissa, exponent = s.lower().split("e")
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{int(exponent):+03d}".replace("E+", "E")
    if s.endswith(".0"):
        s = s[:-1]
    return s


def triple(v):
    return "(" + ",".join(real(c) for c in v) + ")"


class Step:
    def __init__(self, name):
        self.name = name
        self.lines = []

    def add(self, text):
        self.lines.append(text)
        return f"#{len(self.lines)}"

    # Geometry primitives -------------------------------------------------------------------

    def point(self, p):
        return self.add(f"CARTESIAN_POINT('',{triple(p)})")

    def direction(self, d):
        return self.add(f"DIRECTION('',{triple(d)})")

    def axis(self, origin, z=(0, 0, 1), x=(1, 0, 0)):
        return self.add(f"AXIS2_PLACEMENT_3D('',{self.point(origin)},{self.direction(z)},{self.direction(x)})")

    def vertex(self, p):
        return self.add(f"VERTEX_POINT('',{self.point(p)})")

    def line_edge(self, va, vb, a, b):
        d = [b[i] - a[i] for i in range(3)]
        length = math.sqrt(sum(c * c for c in d))
        u = [c / length for c in d]
        vector = self.add(f"VECTOR('',{self.direction(u)},{real(length)})")
        line = self.add(f"LINE('',{self.point(a)},{vector})")
        return self.add(f"EDGE_CURVE('',{va},{vb},{line},.T.)")

    def circle_edge(self, v, center, radius):
        circle = self.add(f"CIRCLE('',{self.axis(center)},{real(radius)})")
        return self.add(f"EDGE_CURVE('',{v},{v},{circle},.T.)")

    def face(self, surface, edges):
        oriented = ",".join(self.add(f"ORIENTED_EDGE('',*,*,{e},{'.T.' if forward else '.F.'})") for e, forward in edges)
        loop = self.add(f"EDGE_LOOP('',({oriented}))")
        bound = self.add(f"FACE_OUTER_BOUND('',{loop},.T.)")
        return self.add(f"ADVANCED_FACE('',({bound}),{surface},.T.)")

    def solid(self, name, faces):
        shell = self.add(f"CLOSED_SHELL('',({','.join(faces)}))")
        return self.add(f"MANIFOLD_SOLID_BREP('{name}',{shell})")

    def box(self, name, lo, hi):
        xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
        corner = {(i, j, k): (xs[i], ys[j], zs[k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        vertex = {key: self.vertex(p) for key, p in corner.items()}
        edges = {}

        def edge(a, b):
            if (b, a) in edges:
                return edges[(b, a)], False
            if (a, b) not in edges:
                edges[(a, b)] = self.line_edge(vertex[a], vertex[b], corner[a], corner[b])
            return edges[(a, b)], True

        # Each loop runs counter-clockwise seen from outside, so the plane normal points out.
        loops = [
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ]
        faces = []
        for normal, keys in loops:
            ref = (0, 0, 1) if normal[2] == 0 else (1, 0, 0)
            plane = self.add(f"PLANE('',{self.axis(corner[keys[0]], normal, ref)})")
            faces.append(self.face(plane, [edge(keys[n], keys[(n + 1) % 4]) for n in range(4)]))
        return self.solid(name, faces)

    def cylinder(self, name, center, radius, z0, z1):
        cx, cy = center
        bottom, top = (cx, cy, z0), (cx, cy, z1)
        vb = self.vertex((cx + radius, cy, z0))
        vt = self.vertex((cx + radius, cy, z1))
        rim_bottom = self.circle_edge(vb, bottom, radius)
        rim_top = self.circle_edge(vt, top, radius)
        seam = self.line_edge(vb, vt, (cx + radius, cy, z0), (cx + radius, cy, z1))
        side_surface = self.add(f"CYLINDRICAL_SURFACE('',{self.axis(bottom)},{real(radius)})")
        side = self.face(side_surface, [(rim_bottom, True), (seam, True), (rim_top, False), (seam, False)])
        top_face = self.face(self.add(f"PLANE('',{self.axis(top)})"), [(rim_top, True)])
        bottom_face = self.face(self.add(f"PLANE('',{self.axis(bottom, (0, 0, -1))})"), [(rim_bottom, False)])
        return self.solid(name, [side, top_face, bottom_face])

    # Product structure ---------------------------------------------------------------------

    def context(self):
        mm = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
        rad = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        sr = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        unc = self.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),{mm},'distance_accuracy_value','confusion accuracy')")
        return self.add(
            "( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(("
            f"{unc})) GLOBAL_UNIT_ASSIGNED_CONTEXT(({mm},{rad},{sr})) "
            "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )")

    def application(self):
        app = self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
        self.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,{app})")
        self.product_context = self.add(f"PRODUCT_CONTEXT('',{app},'mechanical')")
        self.definition_context = self.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{app},'design')")

    def product(self, name):
        product = self.add(f"PRODUCT('{name}','{name}','',({self.product_context}))")
        self.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({product}))")
        formation = self.add(f"PRODUCT_DEFINITION_FORMATION('','',{product})")
        definition = self.add(f"PRODUCT_DEFINITION('design','',{formation},{self.definition_context})")
        shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',{definition})")
        return definition, shape

    def colour(self, item, rgb, styled):
        colour = self.add(f"COLOUR_RGB('',{real(rgb[0])},{real(rgb[1])},{real(rgb[2])})")
        fill_colour = self.add(f"FILL_AREA_STYLE_COLOUR('',{colour})")
        fill = self.add(f"FILL_AREA_STYLE('',({fill_colour}))")
        fill_area = self.add(f"SURFACE_STYLE_FILL_AREA({fill})")
        side = self.add(f"SURFACE_SIDE_STYLE('',({fill_area}))")
        usage = self.add(f"SURFACE_STYLE_USAGE(.BOTH.,{side})")
        assignment = self.add(f"PRESENTATION_STYLE_ASSIGNMENT(({usage}))")
        styled.append(self.add(f"STYLED_ITEM('color',({assignment}),{item})"))

    def text(self):
        header = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION(('Splatelier STEP import test fixture'),'2;1');",
            f"FILE_NAME('{self.name}.step','2026-11-03T00:00:00',(''),(''),'make_step_fixtures.py','','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
            "ENDSEC;",
            "DATA;",
        ]
        body = [f"#{n + 1}={line};" for n, line in enumerate(self.lines)]
        return "\n".join(header + body + ["ENDSEC;", "END-ISO-10303-21;", ""])


WOOD = (0.72, 0.53, 0.34)
DARK_WOOD = (0.36, 0.24, 0.15)
STEEL = (0.25, 0.25, 0.27)
WHITE = (0.92, 0.91, 0.88)


def single_part(name, bodies):
    """One part holding several solids: `bodies` is a list of (solid builder, colour)."""
    s = Step(name)
    s.application()
    definition, shape = s.product(name)
    ctx = s.context()
    solids, styled = [], []
    for build, rgb in bodies:
        solid = build(s)
        solids.append(solid)
        s.colour(solid, rgb, styled)
    rep = s.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',({s.axis((0, 0, 0))},{','.join(solids)}),{ctx})")
    s.add(f"SHAPE_DEFINITION_REPRESENTATION({shape},{rep})")
    s.add(f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',({','.join(styled)}),{ctx})")
    return s


def table():
    top = (1200, 700, 30)
    leg_r, leg_h, inset = 25, 690, 70
    bodies = [(lambda s: s.box("top", (0, 0, leg_h), (top[0], top[1], leg_h + top[2])), WOOD)]
    for x in (inset, top[0] - inset):
        for y in (inset, top[1] - inset):
            bodies.append((lambda s, x=x, y=y: s.cylinder("leg", (x, y), leg_r, 0, leg_h), STEEL))
    return single_part("table", bodies)


def chair():
    seat, seat_z, leg = (450, 450, 25), 430, 35
    bodies = [(lambda s: s.box("seat", (0, 0, seat_z), (seat[0], seat[1], seat_z + seat[2])), WOOD)]
    for x in (0, seat[0] - leg):
        for y in (0, seat[1] - leg):
            height = 900 if y > 0 else seat_z
            bodies.append((lambda s, x=x, y=y, h=height: s.box("leg", (x, y, 0), (x + leg, y + leg, h)), DARK_WOOD))
    for z in (620, 780):
        bodies.append((lambda s, z=z: s.box("rail", (leg, seat[1] - leg, z), (seat[0] - leg, seat[1], z + 70)), WOOD))
    return single_part("chair", bodies)


def side_table():
    bodies = [
        (lambda s: s.cylinder("top", (0, 0), 300, 520, 545), WHITE),
        (lambda s: s.cylinder("post", (0, 0), 30, 15, 520), STEEL),
        (lambda s: s.cylinder("base", (0, 0), 200, 0, 15), STEEL),
    ]
    return single_part("side_table", bodies)


def bookshelf():
    """An assembly: side panels, one shelf part placed five times, and a back panel."""
    width, depth, height, board, back = 800, 300, 1800, 18, 5
    s = Step("bookshelf")
    s.application()
    ctx = s.context()
    styled = []

    def part(name, size, rgb):
        definition, shape = s.product(name)
        origin = s.axis((0, 0, 0))
        solid = s.box(name, (0, 0, 0), size)
        s.colour(solid, rgb, styled)
        rep = s.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',({origin},{solid}),{ctx})")
        s.add(f"SHAPE_DEFINITION_REPRESENTATION({shape},{rep})")
        return definition, rep, origin

    root_definition, root_shape = s.product("bookshelf")
    side = part("side_panel", (board, depth, height), WOOD)
    shelf = part("shelf", (width - 2 * board, depth - back, board), WOOD)
    back_panel = part("back_panel", (width - 2 * board, back, height), WHITE)

    placements = [s.axis((0, 0, 0))]
    usages = []
    instances = [(side, (0, 0, 0)), (side, (width - board, 0, 0))]
    shelf_count = 5
    for n in range(shelf_count):
        z = n * (height - board) / (shelf_count - 1)
        instances.append((shelf, (board, back, z)))
    instances.append((back_panel, (board, 0, 0)))
    for n, ((definition, rep, origin), position) in enumerate(instances):
        placement = s.axis(position)
        placements.append(placement)
        usages.append((n + 1, definition, rep, origin, placement))

    root_rep = s.add(f"SHAPE_REPRESENTATION('bookshelf',({','.join(placements)}),{ctx})")
    s.add(f"SHAPE_DEFINITION_REPRESENTATION({root_shape},{root_rep})")
    for n, definition, rep, origin, placement in usages:
        usage = s.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{n}','instance {n}','',{root_definition},{definition},$)")
        transform = s.add(f"ITEM_DEFINED_TRANSFORMATION('','',{origin},{placement})")
        relation = s.add(
            f"( REPRESENTATION_RELATIONSHIP('','',{rep},{root_rep}) "
            f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION({transform}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
        usage_shape = s.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',{usage})")
        s.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION({relation},{usage_shape})")
    s.add(f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',({','.join(styled)}),{ctx})")
    return s


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    for step in (table(), chair(), side_table(), bookshelf()):
        path = out / f"{step.name}.step"
        path.write_text(step.text())
        print(f"wrote {path} ({len(step.lines)} entities)")


if __name__ == "__main__":
    main()
