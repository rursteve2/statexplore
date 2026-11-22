; Inno Setup script: compile after packaging\build.py has produced dist\StatExplore
[Setup]
LicenseFile=..\LICENSE
AppName=StatExplore
AppVersion=1.1.0
AppPublisher=Stephen
AppPublisherURL=https://github.com/rursteve2/statexplore
OutputBaseFilename=StatExplore-1.1.0-Setup
DefaultDirName={autopf}\StatExplore
DefaultGroupName=StatExplore
OutputDir=..\dist\installer
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
