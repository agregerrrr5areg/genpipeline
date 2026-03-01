"""
FeaturePython object: GenDesignPreservedRegion
==============================================
Wraps a geometry object (Box, Cylinder, or any Shape) to mark it as a 
"Preserved Region" (Non-Design Domain) for the topology optimizer.

The voxel grid corresponding to this shape's bounding box (or volume)
will be locked to density=1.0.
"""

import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets

class PreservedRegionObject:
    def __init__(self, obj):
        obj.Proxy = self
        obj.addProperty("App::PropertyLink", "RegionShape", "Base", "Geometry defining the preserved region")
        obj.addProperty("App::PropertyString", "Label", "Base", "User label")

    def execute(self, obj):
        pass

    def onChanged(self, obj, prop):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None

class PreservedRegionViewProvider:
    def __init__(self, vobj):
        vobj.Proxy = self

    def getIcon(self):
        return "" # Standard icon or path

    def attach(self, vobj):
        self.vobj = vobj
        # Make the linked shape semi-transparent green to indicate "safe zone"
        # This requires accessing the linked shape's ViewObject, which might not always work directly
        # depending on how the user set it up. We'll leave visual styling to the user for now.

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        return None

class AddPreservedRegionPanel:
    def __init__(self):
        self.form = QtWidgets.QWidget()
        self.form.setWindowTitle("Add Preserved Region")
        layout = QtWidgets.QVBoxLayout(self.form)

        layout.addWidget(QtWidgets.QLabel("Select a shape (Box, Cylinder, Body) to mark as preserved."))
        layout.addWidget(QtWidgets.QLabel("The optimizer will keep this volume 100% solid."))
        
        self.info = QtWidgets.QLabel("")
        layout.addWidget(self.info)

        self.btn = QtWidgets.QPushButton("Mark Selected as Preserved")
        self.btn.clicked.connect(self.accept)
        layout.addWidget(self.btn)

    def accept(self):
        import FreeCAD as App
        sel = FreeCADGui.Selection.getSelection()
        
        if not sel:
            QtWidgets.QMessageBox.warning(self.form, "No selection", "Select a shape object first.")
            return

        doc = App.ActiveDocument
        doc.openTransaction("Add Preserved Region")
        
        for s in sel:
            # Create the marker object
            obj = doc.addObject("App::FeaturePython", "PreservedRegion")
            PreservedRegionObject(obj)
            PreservedRegionViewProvider(obj.ViewObject)
            obj.RegionShape = s
            obj.Label = f"Preserved_{s.Name}"
            
            # Optional: Style the original object to look "preserved" (Green transparency)
            if hasattr(s, "ViewObject") and s.ViewObject:
                s.ViewObject.ShapeColor = (0.2, 0.8, 0.2)
                s.ViewObject.Transparency = 50
        
        doc.commitTransaction()
        doc.recompute()
        FreeCADGui.Control.closeDialog()

    def reject(self):
        FreeCADGui.Control.closeDialog()
