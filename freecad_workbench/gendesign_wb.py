"""GenDesign workbench definition — toolbars, menus, icon."""

import FreeCADGui


class GenDesignWorkbench(FreeCADGui.Workbench):
    MenuText = "GenDesign"
    ToolTip  = "Generative design: constraints, loads, VAE optimisation"
    Icon     = ""          # TODO: embed SVG icon path here

    def Initialize(self):
        import commands  # noqa: F401  (registers all Gui.addCommand calls)
        self.appendToolbar("GenDesign", [
            "GenDesign_AddConstraint",
            "GenDesign_AddLoad",
            "GenDesign_SetSeedPart",
            "Separator",
            "GenDesign_ExportConfig",
            "GenDesign_RunOptimisation",
            "GenDesign_ImportResult",
        ])
        self.appendMenu("GenDesign", [
            "GenDesign_AddConstraint",
            "GenDesign_AddLoad",
            "GenDesign_SetSeedPart",
            "Separator",
            "GenDesign_ExportConfig",
            "GenDesign_RunOptimisation",
            "GenDesign_ImportResult",
        ])

    def Activated(self):
        pass

    def Deactivated(self):
        pass

    def GetClassName(self):
        return "Gui::PythonWorkbench"
