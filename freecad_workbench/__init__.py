"""
GenDesign FreeCAD Workbench
===========================
Generative design configuration UI for the genpipeline VAE+BO system.

Lets users define:
  - Fixed faces / symmetry planes (constraints)
  - Point loads and distributed pressure (loads)
  - Seed geometry and optimisation parameters

Exports a structured JSON that freecad_bridge.py picks up automatically,
and provides a "Run Optimisation" button that shells back to the WSL2 pipeline.

Install:
  Copy (or symlink) this directory to FreeCAD's Mod folder:
    %APPDATA%\..\Local\Programs\FreeCAD 1.0\Mod\GenDesign\
"""

import FreeCAD
import FreeCADGui

from gendesign_wb import GenDesignWorkbench

FreeCADGui.addWorkbench(GenDesignWorkbench)
