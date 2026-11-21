"""PyInstaller entry point (absolute import: a frozen __main__ has no parent package)."""
import sys

from statexplore.__main__ import main

sys.exit(main())
