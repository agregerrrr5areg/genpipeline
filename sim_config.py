# Simulation Physical Environment Configuration (Point-Based)
# ===========================================================

SIM_PHYSICS = {
    # 1. FORCE / LOAD
    "force_n": 1000.0,
    "force_direction": [0, 0, -1],
    
    # 2. COORDINATE-BASED TARGETING (In mm)
    # Set these to None to use Face-based normals instead.
    "load_point": [100.0, 10.0, 5.0], # Apply force at the tip (X=100, Center)
    "fixed_point": [0.0, 10.0, 5.0],  # Anchor at the root (X=0, Center)
    
    # 3. CONSTRAINTS (FALLBACK)
    "fixed_face_normal": [-1, 0, 0],
    "load_face_normal": [1, 0, 0],
}
