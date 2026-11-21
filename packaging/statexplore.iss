; Inno Setup script: compile after packaging\build.py has produced dist\StatExplore
[Setup]
AppName=StatExplore
AppVersion=1.0
DefaultDirName={autopf}\StatExplore
DefaultGroupName=StatExplore
OutputDir=..\dist\installer
OutputBaseFilename=StatExplore-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "..\dist\StatExplore\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\StatExplore"; Filename: "{app}\StatExplore.exe"
Name: "{autodesktop}\StatExplore"; Filename: "{app}\StatExplore.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\StatExplore.exe"; Description: "Launch StatExplore"; Flags: nowait postinstall skipifsilent
