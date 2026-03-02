"""
FeaturePython object: GenDesignConstraint
=========================================
Stores a topological constraint: a reference to a face/edge on a body
plus the constraint type (fixed, symmetry, preserve).

Serialised to JSON via export_pipeline.py.
"""

import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets, QtCore


CONSTRAINT_TYPES = ["fixed", "symmetry", "preserve", "mounting"]


# ── Document object (non-GUI, serialised) ─────────────────────────────────────

class ConstraintObject:
    """The non-GUI data object stored in the FreeCAD document."""

    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyString",     "ConstraintType", "Constraint", "Type of constraint")
        obj.addProperty("App::PropertyLinkSubList", "References",    "Constraint", "Face/edge references")
        obj.addProperty("App::PropertyString",      "Label",         "Constraint", "User label")
        obj.ConstraintType = "fixed"

    def execute(self, obj):
        pass

    def onChanged(self, obj, prop):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None


class ConstraintViewProvider:
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

class AddConstraintPanel:
    """Task panel for adding a constraint to the document."""

    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("Add Constraint")
        layout = QtWidgets.QVBoxLayout(self.form)

        layout.addWidget(QtWidgets.QLabel("Constraint type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(CONSTRAINT_TYPES)
        layout.addWidget(self.type_combo)

        layout.addWidget(QtWidgets.QLabel("Select faces in the 3D view, then click OK."))
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        # Show currently selected faces
        self.face_list = QtWidgets.QListWidget()
        layout.addWidget(self.face_list)
        self._refresh_selection()

        # Refresh button
        btn = QtWidgets.QPushButton("Refresh selection")
        btn.clicked.connect(self._refresh_selection)
        layout.addWidget(btn)

    def _refresh_selection(self):
        self.face_list.clear()
        sel = FreeCADGui.Selection.getSelectionEx()
        for s in sel:
            for sub in s.SubElementNames:
                self.face_list.addItem(f"{s.ObjectName}.{sub}")

    def accept(self):
        import FreeCAD as App
        sel = FreeCADGui.Selection.getSelectionEx()
        refs = []
        for s in sel:
            refs.append((s.Object, s.SubElementNames))

        if not refs:
            QtWidgets.QMessageBox.warning(self.form, "No selection",
                                          "Select at least one face before clicking OK.")
            return False

        doc = App.ActiveDocument
        obj = doc.addObject("App::FeaturePython", "Constraint")
        ConstraintObject(obj)
        ConstraintViewProvider(obj.ViewObject)
        obj.ConstraintType = self.type_combo.currentText()
        obj.References = refs
        obj.Label = f"Constraint_{self.type_combo.currentText()}"
        doc.recompute()
        FreeCADGui.Control.closeDialog()
        return True

    def reject(self):
        FreeCADGui.Control.closeDialog()


# ── Factory ───────────────────────────────────────────────────────────────────

def make_constraint(doc, constraint_type: str, references: list, label: str = None):
    """Programmatically create a constraint (used by export/test code)."""
    obj = doc.addObject("App::FeaturePython", "Constraint")
    ConstraintObject(obj)
    ConstraintViewProvider(obj.ViewObject)
    obj.ConstraintType = constraint_type
    obj.References = references
    obj.Label = label or f"Constraint_{constraint_type}"
    return obj
