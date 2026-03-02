"""
FeaturePython object: GenDesignLoad
====================================
Stores a mechanical load applied to a face/vertex:
  - Type: force | pressure | acceleration
  - Magnitude (N, MPa, or m/s²)
  - Direction vector
  - Face/vertex references
"""

import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets, QtCore


LOAD_TYPES = ["force", "pressure", "acceleration"]


# ── Document object ───────────────────────────────────────────────────────────

class LoadObject:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyString",      "LoadType",   "Load", "force | pressure | acceleration")
        obj.addProperty("App::PropertyFloat",       "Magnitude",  "Load", "Magnitude (N / MPa / m·s⁻²)")
        obj.addProperty("App::PropertyVector",      "Direction",  "Load", "Direction unit vector")
        obj.addProperty("App::PropertyLinkSubList", "References", "Load", "Target faces / vertices")
        obj.LoadType  = "force"
        obj.Magnitude = 1000.0
        obj.Direction = FreeCAD.Vector(0, 0, -1)

    def execute(self, obj):
        pass

    def onChanged(self, obj, prop):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class LoadViewProvider:
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


# ── Task panel ────────────────────────────────────────────────────────────────

class AddLoadPanel:
    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("Add Load")
        layout = QtWidgets.QVBoxLayout(self.form)

        layout.addWidget(QtWidgets.QLabel("Load type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(LOAD_TYPES)
        layout.addWidget(self.type_combo)

        layout.addWidget(QtWidgets.QLabel("Magnitude:"))
        self.mag_spin = QtWidgets.QDoubleSpinBox()
        self.mag_spin.setRange(0.0, 1e9)
        self.mag_spin.setValue(1000.0)
        self.mag_spin.setSuffix(" N / MPa / m·s⁻²")
        layout.addWidget(self.mag_spin)

        layout.addWidget(QtWidgets.QLabel("Direction (X, Y, Z):"))
        dir_widget = QtWidgets.QWidget()
        dir_layout = QtWidgets.QHBoxLayout(dir_widget)
        self.dx = QtWidgets.QDoubleSpinBox(); self.dx.setRange(-1,1); self.dx.setSingleStep(0.1); self.dx.setValue(0)
        self.dy = QtWidgets.QDoubleSpinBox(); self.dy.setRange(-1,1); self.dy.setSingleStep(0.1); self.dy.setValue(0)
        self.dz = QtWidgets.QDoubleSpinBox(); self.dz.setRange(-1,1); self.dz.setSingleStep(0.1); self.dz.setValue(-1)
        dir_layout.addWidget(self.dx); dir_layout.addWidget(self.dy); dir_layout.addWidget(self.dz)
        layout.addWidget(dir_widget)

        layout.addWidget(QtWidgets.QLabel("Select target faces, then click OK."))
        self.face_list = QtWidgets.QListWidget()
        layout.addWidget(self.face_list)
        btn = QtWidgets.QPushButton("Refresh selection")
        btn.clicked.connect(self._refresh_selection)
        layout.addWidget(btn)
        self._refresh_selection()

    def _refresh_selection(self):
        self.face_list.clear()
        for s in FreeCADGui.Selection.getSelectionEx():
            for sub in s.SubElementNames:
                self.face_list.addItem(f"{s.ObjectName}.{sub}")

    def accept(self):
        import FreeCAD as App
        sel = FreeCADGui.Selection.getSelectionEx()
        refs = [(s.Object, s.SubElementNames) for s in sel]
        if not refs:
            QtWidgets.QMessageBox.warning(self.form, "No selection",
                                          "Select at least one face before clicking OK.")
            return False

        doc = App.ActiveDocument
        obj = doc.addObject("App::FeaturePython", "Load")
        LoadObject(obj)
        LoadViewProvider(obj.ViewObject)
        obj.LoadType  = self.type_combo.currentText()
        obj.Magnitude = self.mag_spin.value()
        obj.Direction = App.Vector(self.dx.value(), self.dy.value(), self.dz.value())
        obj.References = refs
        obj.Label = f"Load_{self.type_combo.currentText()}_{obj.Magnitude:.0f}"
        doc.recompute()
        FreeCADGui.Control.closeDialog()
        return True

    def reject(self):
        FreeCADGui.Control.closeDialog()


def make_load(doc, load_type, magnitude, direction, references, label=None):
    obj = doc.addObject("App::FeaturePython", "Load")
    LoadObject(obj)
    LoadViewProvider(obj.ViewObject)
    obj.LoadType  = load_type
    obj.Magnitude = magnitude
    obj.Direction = FreeCAD.Vector(*direction)
    obj.References = references
    obj.Label = label or f"Load_{load_type}"
    return obj
