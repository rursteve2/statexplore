# PyInstaller spec — build with:  .venv\Scripts\python packaging\build.py
from pathlib import Path

root = Path(SPECPATH).parent
a = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root / "src")],
    datas=[(str(root / "src" / "statexplore" / "r" / "worker.R"), "statexplore/r"),
           (str(root / "src" / "statexplore" / "r" / "console_worker.R"), "statexplore/r")],
    hiddenimports=["pyreadstat._readstat_parser", "openpyxl", "docx", "pyarrow", "statexplore.gui.main_window"],
    excludes=["tkinter", "pytest", "statsmodels", "matplotlib"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="StatExplore", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="StatExplore")
