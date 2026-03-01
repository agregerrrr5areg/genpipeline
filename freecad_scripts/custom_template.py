"""
USER TEMPLATE: Define your custom geometry here.
The AI will call 'make_custom_shape' with a dictionary of parameters.
"""

def make_custom_shape(params):
    import FreeCAD as App
    import Part
    import Sketcher

    # 1. Extract parameters the AI is optimizing
    h = params.get("h_mm", 10.0)
    r = params.get("r_mm", 3.0)
    
    # --- YOUR CUSTOM FREECAD CODE START ---
    # Example: A beam with a rounded top and custom hole
    
    # Create main body
    base = Part.makeBox(100.0, 20.0, h)
    
    # Add a custom feature (e.g., a cylinder on top)
    cyl_top = Part.makeCylinder(5.0, 20.0, App.Vector(50, 0, h), App.Vector(0, 1, 0))
    body = base.fuse(cyl_top)
    
    # Cut the hole the AI is optimizing
    if r > 0.5:
        hole = Part.makeCylinder(r, 20.0, App.Vector(50, 0, h/2.0), App.Vector(0, 1, 0))
        final_shape = body.cut(hole)
    else:
        final_shape = body
        
    # --- YOUR CUSTOM FREECAD CODE END ---
    
    return final_shape
