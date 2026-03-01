"""
FeaturePython object: GenDesignSeedPart
========================================
Marks a body as the seed geometry for optimisation.
Stores:
  - Reference to the source body
  - Geometry type (cantilever | lbracket | tapered | ribbed)
  - Optimisation parameters (volume fraction, manufacturing flags)
  - WSL2 pipeline path for the Run button
"""

import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets


GEOMETRY_TYPES = ["cantilever", "lbracket", "tapered", "ribbed"]


class SeedPartObject:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyLink",   "SourceBody",      "Seed", "Body to optimise")
        obj.addProperty("App::PropertyString", "GeometryType",    "Seed", "cantilever | lbracket | tapered | ribbed")
        obj.addProperty("App::PropertyFloat",  "VolumeFraction",  "Seed", "Target volume fraction (0-1)")
        obj.addProperty("App::PropertyFloat",  "MaxStressMPa",    "Seed", "Stress limit MPa")
        obj.addProperty("App::PropertyBool",   "NoOverhang",      "Seed", "Enforce no-overhang (45 deg)")
        obj.addProperty("App::PropertyString", "WSL2PipelinePath","Seed", "Path to genpipeline in WSL2")
        obj.addProperty("App::PropertyString", "CheckpointPath",  "Seed", "VAE checkpoint .pth path (WSL2)")
        obj.addProperty("App::PropertyInteger","NIter",           "Seed", "Optimisation iterations")
        obj.GeometryType     = "cantilever"
        obj.VolumeFraction   = 0.4
        obj.MaxStressMPa     = 250.0
        obj.NoOverhang       = False
        obj.WSL2PipelinePath = "/home/genpipeline"
        obj.CheckpointPath   = "/home/genpipeline/checkpoints/vae_best.pth"
        obj.NIter            = 50

    def execute(self, obj):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class SeedPartViewProvider:
    def __init__(self, vobj):
        vobj.Proxy = self

    def getIcon(self):
        return ""

    def attach(self, vobj):
        self.vobj = vobj

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class SetSeedPartPanel:
    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("Set Seed Part")
        layout = QtWidgets.QVBoxLayout(self.form)

        layout.addWidget(QtWidgets.QLabel("Select the body to optimise, then fill in parameters."))

        layout.addWidget(QtWidgets.QLabel("Geometry type:"))
        self.geom_combo = QtWidgets.QComboBox()
        self.geom_combo.addItems(GEOMETRY_TYPES)
        layout.addWidget(self.geom_combo)

        layout.addWidget(QtWidgets.QLabel("Volume fraction target (0–1):"))
        self.vf_spin = QtWidgets.QDoubleSpinBox()
        self.vf_spin.setRange(0.05, 1.0); self.vf_spin.setSingleStep(0.05); self.vf_spin.setValue(0.4)
        layout.addWidget(self.vf_spin)

        layout.addWidget(QtWidgets.QLabel("Max stress (MPa):"))
        self.stress_spin = QtWidgets.QDoubleSpinBox()
        self.stress_spin.setRange(1.0, 5000.0); self.stress_spin.setValue(250.0)
        layout.addWidget(self.stress_spin)

        layout.addWidget(QtWidgets.QLabel("Optimisation iterations:"))
        self.iter_spin = QtWidgets.QSpinBox()
        self.iter_spin.setRange(5, 2000); self.iter_spin.setValue(50)
        layout.addWidget(self.iter_spin)

        layout.addWidget(QtWidgets.QLabel("WSL2 pipeline path:"))
        self.wsl_edit = QtWidgets.QLineEdit("/home/genpipeline")
        layout.addWidget(self.wsl_edit)

        layout.addWidget(QtWidgets.QLabel("VAE checkpoint (WSL2 path):"))
        self.ckpt_edit = QtWidgets.QLineEdit("/home/genpipeline/checkpoints/vae_best.pth")
        layout.addWidget(self.ckpt_edit)

        self.overhang_cb = QtWidgets.QCheckBox("Enforce no-overhang (45°)")
        layout.addWidget(self.overhang_cb)

    def accept(self):
        import FreeCAD as App
        sel = FreeCADGui.Selection.getSelection()
        doc = App.ActiveDocument

        obj = doc.addObject("App::FeaturePython", "SeedPart")
        SeedPartObject(obj)
        SeedPartViewProvider(obj.ViewObject)
        if sel:
            obj.SourceBody = sel[0]
        obj.GeometryType     = self.geom_combo.currentText()
        obj.VolumeFraction   = self.vf_spin.value()
        obj.MaxStressMPa     = self.stress_spin.value()
        obj.NIter            = self.iter_spin.value()
        obj.WSL2PipelinePath = self.wsl_edit.text()
        obj.CheckpointPath   = self.ckpt_edit.text()
        obj.NoOverhang       = self.overhang_cb.isChecked()
        obj.Label = f"SeedPart_{obj.GeometryType}"
        doc.recompute()
        FreeCADGui.Control.closeDialog()
        return True

    def reject(self):
        FreeCADGui.Control.closeDialog()
