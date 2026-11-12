Set-Location $PSScriptRoot
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $ProjectPython)) { $ProjectPython = 'E:\yolo_course\.venv\Scripts\python.exe' }
if (Test-Path $ProjectPython) { & $ProjectPython app.py } else { py -3.12 app.py }
