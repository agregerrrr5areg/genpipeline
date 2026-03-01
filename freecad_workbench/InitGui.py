# GenDesign FreeCAD Workbench initialization
import sys
import os
import FreeCAD
import FreeCADGui
from PySide2 import QtCore, QtWidgets

# --- ROBUST LOGGING ---
LOG_FILE = os.path.join(os.path.expanduser("~"), "Documents", "gendesign_debug.log")

def log_message(msg, is_error=False):
    timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
    prefix = "[ERROR]" if is_error else "[INFO]"
    line = f"{timestamp} {prefix} {msg}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except:
        pass
    if is_error:
        FreeCAD.Console.PrintError(line)
    else:
        FreeCAD.Console.PrintMessage(line)

# 1. ROBUST PATH DETECTION
def get_workbench_path():
    try:
        user_mod = os.path.join(FreeCAD.ConfigGet("UserModDir"), "GenDesign")
        if os.path.exists(user_mod):
            return user_mod
        system_mod = os.path.join(FreeCAD.ConfigGet("AppHomePath"), "Mod", "GenDesign")
        if os.path.exists(system_mod):
            return system_mod
    except Exception as e:
        log_message(f"Path detection error: {e}", True)
    return None

WB_PATH = get_workbench_path()
if WB_PATH and WB_PATH not in sys.path:
    sys.path.append(WB_PATH)
    log_message(f"Added {WB_PATH} to sys.path")

class GenDesignWorkbench(FreeCADGui.Workbench):
    MenuText = "&GenDesign"
    ToolTip = "Generative design: constraints, loads, VAE optimization"
    
    def Initialize(self):
        try:
            import commands
            self.appendToolbar("GenDesign", ["GenDesign_AddConstraint", "GenDesign_AddLoad", "GenDesign_AddPreservedRegion", "GenDesign_SetSeedPart", "Separator", "GenDesign_ExportConfig", "GenDesign_RunOptimisation", "GenDesign_ImportResult"])
            self.appendMenu("&GenDesign", ["GenDesign_AddConstraint", "GenDesign_AddLoad", "GenDesign_AddPreservedRegion", "GenDesign_SetSeedPart", "Separator", "GenDesign_ExportConfig", "GenDesign_RunOptimisation", "GenDesign_ImportResult"])
            log_message("Workbench activated.")
        except Exception as e:
            log_message(f"Workbench Init Failure: {e}", True)

    def GetClassName(self): return "Gui::PythonWorkbench"

FreeCADGui.addWorkbench(GenDesignWorkbench())

# 2. GLOBAL MENU INJECTION
def inject_global_menu():
    try:
        mw = FreeCADGui.getMainWindow()
        if not mw:
            QtCore.QTimer.singleShot(500, inject_global_menu)
            return
            
        menubar = mw.menuBar()
        for action in menubar.actions():
            if "GenDesign" in action.text(): return

        import commands
        gd_menu = menubar.addMenu("&GenDesign")
        menu_items = [
            ("Add Constraint", "GenDesign_AddConstraint"),
            ("Add Load", "GenDesign_AddLoad"),
            ("Add Preserved Region", "GenDesign_AddPreservedRegion"),
            ("Set Seed Part", "GenDesign_SetSeedPart"),
            (None, None),
            ("Export Configuration", "GenDesign_ExportConfig"),
            ("Run Optimization Pipeline", "GenDesign_RunOptimisation"),
            ("Import Results", "GenDesign_ImportResult"),
        ]
        for label, cmd_name in menu_items:
            if label is None: gd_menu.addSeparator()
            else:
                action = gd_menu.addAction(label)
                # Catch the boolean argument from 'triggered' to prevent TypeError
                action.triggered.connect(lambda checked=False, name=cmd_name: FreeCADGui.runCommand(name))
        log_message("Top-Level Menu injected.")
    except Exception as e:
        log_message(f"Menu Injection Error: {e}", True)

QtCore.QTimer.singleShot(2500, inject_global_menu)
