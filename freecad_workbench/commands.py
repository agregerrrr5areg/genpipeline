"""
FreeCAD Gui commands — registered with FreeCADGui.addCommand().
Each class maps to one toolbar/menu entry.
"""

import subprocess
import os
import FreeCAD
import FreeCADGui
from PySide2 import QtWidgets, QtCore


# ── Add Constraint ────────────────────────────────────────────────────────────

class CmdAddConstraint:
    def GetResources(self):
        return {"MenuText": "Add Constraint", "ToolTip": "Add a fixed/symmetry/preserve constraint"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from constraint_obj import AddConstraintPanel
        FreeCADGui.Control.showDialog(AddConstraintPanel())


FreeCADGui.addCommand("GenDesign_AddConstraint", CmdAddConstraint())


# ── Add Load ─────────────────────────────────────────────────────────────────

class CmdAddLoad:
    def GetResources(self):
        return {"MenuText": "Add Load", "ToolTip": "Add a force/pressure/acceleration load"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from load_obj import AddLoadPanel
        FreeCADGui.Control.showDialog(AddLoadPanel())


FreeCADGui.addCommand("GenDesign_AddLoad", CmdAddLoad())


# ── Set Seed Part ─────────────────────────────────────────────────────────────

class CmdSetSeedPart:
    def GetResources(self):
        return {"MenuText": "Set Seed Part", "ToolTip": "Mark a body as the seed geometry for optimisation"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from seed_part import SetSeedPartPanel
        FreeCADGui.Control.showDialog(SetSeedPartPanel())


FreeCADGui.addCommand("GenDesign_SetSeedPart", CmdSetSeedPart())


# ── Export Config ─────────────────────────────────────────────────────────────

class CmdExportConfig:
    def GetResources(self):
        return {"MenuText": "Export Config", "ToolTip": "Export constraints/loads to gendesign_config.json"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from export_pipeline import export_config
        doc = FreeCAD.ActiveDocument
        cfg = export_config(doc)
        QtWidgets.QMessageBox.information(
            None, "GenDesign",
            f"Config exported.\nGeometry: {cfg['geometry_type']}\n"
            f"Constraints: {len(cfg['constraints'])}\n"
            f"Loads: {len(cfg['loads'])}"
        )


FreeCADGui.addCommand("GenDesign_ExportConfig", CmdExportConfig())


# ── Run Optimisation ──────────────────────────────────────────────────────────

class CmdRunOptimisation:
    """
    Exports config then shells to WSL2 to run the VAE+BO pipeline.
    Streams stdout to a FreeCAD progress dialog.
    """

    def GetResources(self):
        return {"MenuText": "Run Optimisation", "ToolTip": "Run VAE+BO optimisation via WSL2 pipeline"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from export_pipeline import export_config, find_seed_part
        doc = FreeCAD.ActiveDocument
        seed = find_seed_part(doc)

        # Export config to Windows Temp so WSL2 can read it
        config_win  = "C:\\Windows\\Temp\\gendesign_config.json"
        config_wsl  = "/mnt/c/Windows/Temp/gendesign_config.json"
        export_config(doc, output_path=config_win)

        pipeline_path = seed.WSL2PipelinePath if seed else "/home/genpipeline"
        ckpt_path     = seed.CheckpointPath   if seed else "/home/genpipeline/checkpoints/vae_best.pth"
        n_iter        = seed.NIter             if seed else 50

        # Build WSL2 command
        wsl_cmd = (
            f"cd {pipeline_path} && "
            f"source venv/bin/activate && "
            f"python optimization_engine.py "
            f"--model-checkpoint {ckpt_path} "
            f"--n-iterations {n_iter} "
            f"--config-path {config_wsl}"
        )

        dlg = _ProgressDialog("Running optimisation…")
        dlg.show()
        QtCore.QCoreApplication.processEvents()

        try:
            proc = subprocess.Popen(
                ["wsl", "bash", "-c", wsl_cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_lines = []
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line)
                FreeCAD.Console.PrintMessage(f"[GenDesign] {line}\n")
                dlg.append(line)
                QtCore.QCoreApplication.processEvents()
                if dlg.cancelled:
                    proc.terminate()
                    break
            proc.wait()
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(None, "GenDesign",
                                           "wsl.exe not found. Is WSL2 installed?")
        finally:
            dlg.close()

        if proc.returncode == 0:
            QtWidgets.QMessageBox.information(None, "GenDesign",
                                              "Optimisation complete. Use Import Result to load the best design.")
        else:
            QtWidgets.QMessageBox.warning(None, "GenDesign",
                                          f"Pipeline exited with code {proc.returncode}.\nCheck FreeCAD console for details.")


FreeCADGui.addCommand("GenDesign_RunOptimisation", CmdRunOptimisation())


# ── Import Result ─────────────────────────────────────────────────────────────

class CmdImportResult:
    """Import the best_design.stl produced by the optimisation back into FreeCAD."""

    def GetResources(self):
        return {"MenuText": "Import Result", "ToolTip": "Import optimised STL result as a new body"}

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None

    def Activated(self):
        from export_pipeline import find_seed_part
        doc  = FreeCAD.ActiveDocument
        seed = find_seed_part(doc)
        pipeline_path = seed.WSL2PipelinePath if seed else "/home/genpipeline"

        # Convert WSL2 path to Windows path for file dialog default
        wsl_result = f"{pipeline_path}/optimization_results/best_design.stl"
        win_result = _wsl_to_win(wsl_result)

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Import optimised STL", win_result, "STL files (*.stl)"
        )
        if not path:
            return

        import Mesh
        mesh = Mesh.Mesh()
        mesh.read(path)
        mesh_obj = doc.addObject("Mesh::Feature", "OptimisedDesign")
        mesh_obj.Mesh = mesh
        doc.recompute()
        FreeCADGui.SendMsgToActiveView("ViewFit")
        FreeCAD.Console.PrintMessage(f"[GenDesign] Imported {path}\n")


FreeCADGui.addCommand("GenDesign_ImportResult", CmdImportResult())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wsl_to_win(wsl_path: str) -> str:
    if wsl_path.startswith("/mnt/"):
        parts = wsl_path[5:].split("/", 1)
        drive = parts[0].upper()
        rest  = parts[1].replace("/", "\\") if len(parts) > 1 else ""
        return f"{drive}:\\{rest}"
    # /home/... → UNC path via wsl.localhost
    distro = "Ubuntu"
    return f"\\\\wsl.localhost\\{distro}" + wsl_path.replace("/", "\\")


class _ProgressDialog(QtWidgets.QDialog):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self.cancelled = False
        layout = QtWidgets.QVBoxLayout(self)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        btn = QtWidgets.QPushButton("Cancel")
        btn.clicked.connect(self._cancel)
        layout.addWidget(btn)

    def append(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _cancel(self):
        self.cancelled = True
