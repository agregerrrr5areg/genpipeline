#!/usr/bin/env python3
"""
FreeCAD script to properly set up FEM analysis for topology optimization.
This script will:
1. Open an existing FreeCAD file
2. Create complete FEM analysis setup with all required objects
3. Add material properties, loads, and constraints
4. Run the analysis properly
5. Save the results back to the file
"""

import FreeCAD
import Fem
import Part
import os

# Configuration for topology optimization
TOPOLOGY_MATERIAL = "Steel"
TOPOLOGY_LOAD = 1000  # N
TOPOLOGY_CONSTRAINT = "Fixed Support"


def add_complete_fem_analysis(doc_path, analysis_name="TopologyOptimization"):
    """Add complete FEM analysis to a FreeCAD document"""
    # Open the document
    doc = FreeCAD.open(doc_path)
    print(f"Opened document: {doc.Name}")

    # Find the main solid body to analyze
    solid = None
    for obj in doc.Objects:
        if obj.TypeId == "PartDesign::Body":
            solid = obj
            print(f"Found solid body: {obj.Name}")
            break

    if not solid:
        print("No solid body found in document")
        return False

    # Create FEM analysis
    fem_analysis = doc.addObject("Fem::FemAnalysis", analysis_name)
    print(f"Created FEM analysis: {fem_analysis.Name}")

    # Create FEM mesh
    mesh = doc.addObject("Fem::FemMeshGmsh", "Mesh")
    mesh.ElementDimension = "Tetrahedra"
    mesh.SecondOrder = False
    mesh.MaxElementSize = 1.0  # Adjust based on model size
    mesh.Refine = "False"
    mesh.Solid = solid
    print(f"Created FEM mesh: {mesh.Name}")

    # Add material
    material = doc.addObject("Fem::FemMaterial", "Material")
    material.Material = TOPOLOGY_MATERIAL
    material.E = 210e9  # Young's modulus for steel (Pa)
    material.nu = 0.3  # Poisson's ratio
    material.Rho = 7850  # Density (kg/m^3)
    print(f"Added material: {material.Name}")

    # Create fixed constraint
    fixed = doc.addObject("Fem::FemConstraintFixed", "Fixed")
    fixed.References = [(solid, "Face", 1)]  # First face as fixed
    print(f"Added fixed constraint: {fixed.Name}")

    # Create force constraint
    force = doc.addObject("Fem::FemConstraintForce", "Force")
    force.References = [(solid, "Face", 2)]  # Second face as loaded
    force.Force = [0, 0, -TOPOLOGY_LOAD]  # Downward force
    print(f"Added force constraint: {force.Name}")

    # Add all objects to the analysis
    fem_analysis.Objects = [mesh, material, fixed, force]

    # Run the analysis
    print("Running FEM analysis...")
    try:
        fem_analysis.solve()
        print("FEM analysis completed successfully")

        # Save the results
        results = fem_analysis.Results[0]
        print(f"Results saved: {results}")

        # Save the document with FEM analysis
        doc.saveAs(doc_path.replace(".FCStd", "_with_complete_fem.FCStd"))
        print(
            f"Document saved with complete FEM analysis: {doc_path.replace('.FCStd', '_with_complete_fem.FCStd')}"
        )

        return True
    except Exception as e:
        print(f"FEM analysis failed: {e}")
        return False

    finally:
        # Close the document
        FreeCAD.closeDocument(doc.Name)


def main():
    """Main function to process all FreeCAD files"""
    # Get all FCStd files in the directory
    import glob

    # Process all FCStd files in the FreeCAD designs directory
    fcstd_files = glob.glob("/mnt/c/Users/PC-PC/Documents/FreeCAD Designs/*.FCStd")
    print(f"Found {len(fcstd_files)} FreeCAD files to process")

    for fcstd_file in fcstd_files:
        print(f"\nProcessing file: {fcstd_file}")
        success = add_complete_fem_analysis(fcstd_file)
        if success:
            print(f"Successfully added complete FEM analysis to: {fcstd_file}")
        else:
            print(f"Failed to add complete FEM analysis to: {fcstd_file}")

    print("\nAll files processed")


if __name__ == "__main__":
    main()
