# Third-party software in StatExplore releases

StatExplore's own code is MIT-licensed (see LICENSE). The release builds (installer and portable
zip) also **bundle** the following software, unmodified, each under its own license:

| Component | License | Source |
|---|---|---|
| R 4.6.2 (`R\` folder) | GPL-2 / GPL-3 | https://cran.r-project.org/src/base/R-4/ |
| R packages (`rlib\` folder; exact list and versions in `rlib\packages.lock.json`) | Each package's own license, mostly GPL-2/GPL-3, some MIT/BSD (see each package's `DESCRIPTION` file) | https://cran.r-project.org/web/packages/ |
| Python 3.11 runtime | PSF License | https://www.python.org/downloads/source/ |
| Qt 6 / PySide6 (Qt for Python) | LGPL-3.0 (used as dynamically linked libraries) | https://download.qt.io/official_releases/ · https://code.qt.io/cgit/pyside/pyside-setup.git |
| pandas, NumPy, SciPy, pyarrow, openpyxl, python-docx | BSD-3-Clause / BSD / MIT | https://pypi.org (each project's page) |
| pyreadstat (ReadStat) | Apache-2.0 / MIT | https://github.com/Roche/pyreadstat |

Source code for the GPL and LGPL components is available at the links above, at the versions listed.
You may replace the bundled Qt libraries (in `_internal\PySide6`) with your own compatible build, as
permitted by the LGPL.
