"""
USER TEMPLATE: Robust Bridge Geometry (V3)
Defines two 20x20x20 support boxes joined by a solid span.
The span overlaps the boxes to ensure a perfect boolean fuse.
"""

def make_custom_shape(params):
    import FreeCAD as App
    import Part

    # Extract parameters (Height and Thickness of the joining span)
    h = params.get("h_mm", 15.0)
    t = params.get("r_mm", 10.0)
    
    # Scale/Physical Constraints
    # Ensure h doesn't go below a minimum for connectivity
    h = max(h, 2.0)
    t = max(t, 2.0)
    
    box_size = 20.0
    gap = 40.0
    
    # 1. Support Boxes
    # Box 1: [0, 20] x [0, 20] x [0, 20]
    box_l = Part.makeBox(box_size, box_size, box_size, App.Vector(0, 0, 0))
    # Box 2: [60, 80] x [0, 20] x [0, 20]
    box_r = Part.makeBox(box_size, box_size, box_size, App.Vector(box_size + gap, 0, 0))
    
    # 2. Joining Span (The bridge)
    # Start at X=19.0 (1mm overlap) and end at X=61.0 (1mm overlap)
    # Span length = 40 + 2 = 42
    span_len = gap + 2.0
    span_y = (box_size - t) / 2.0
    # Span is centered on Y, and sits on the ground (Z=0) up to height H
    span = Part.makeBox(span_len, t, h, App.Vector(box_size - 1.0, span_y, 0))
    
    # Fuse into a single solid body
    shape = box_l.fuse(box_r).fuse(span)
    
    return shape
