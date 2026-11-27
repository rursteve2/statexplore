; Inno Setup script: compile after packaging\build.py has produced dist\StatExplore
; build.py passes /DDistDir=<build dir>\dist (STATEXPLORE_BUILD_DIR); default: the repo's dist folder
#ifndef DistDir
  #define DistDir "..\dist"
#endif
[Setup]
LicenseFile=..\LICENSE
AppName=StatExplore
AppVersion=1.3.4
AppPublisher=Stephen
AppPublisherURL=https://github.com/rursteve2/statexplore
OutputBaseFilename=StatExplore-1.3.4-Setup
DefaultDirName={autopf}\StatExplore
DefaultGroupName=StatExplore
OutputDir=..\dist\installer
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "{#DistDir}\StatExplore\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\StatExplore"; Filename: "{app}\StatExplore.exe"
Name: "{autodesktop}\StatExplore"; Filename: "{app}\StatExplore.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\StatExplore.exe"; Description: "Launch StatExplore"; Flags: nowait postinstall skipifsilent
