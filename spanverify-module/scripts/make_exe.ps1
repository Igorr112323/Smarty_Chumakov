<#
    Сборка SpanVerify в один исполняемый файл под Windows.

    Запуск (из папки spanverify-module):
        powershell -ExecutionPolicy Bypass -File scripts\make_exe.ps1

    Результат: release\spanverify.exe — двойной клик открывает браузер с
    интерфейсом. Рядом кладутся config\ и data\, их можно править без
    пересборки.
#>

[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$SkipTests,
    [switch]$IncludeHF
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "== SpanVerify: сборка Windows-бинарника ==" -ForegroundColor Cyan
Write-Host "Рабочая папка: $root"

# 1. Проверка версии Python
$version = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Python: $version"
& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Нужен Python 3.10 или новее." }

# 2. Зависимости сборки
Write-Host "Установка зависимостей сборки..." -ForegroundColor Cyan
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
if ($IncludeHF) { & $Python -m pip install -r requirements-hf.txt }

# 3. Тесты (можно пропустить флагом -SkipTests)
if (-not $SkipTests) {
    Write-Host "Прогон тестов..." -ForegroundColor Cyan
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Тесты не прошли — сборка остановлена." }
}

# 4. Калибровка: если файла нет — обучить на демонстрационном корпусе
if (-not (Test-Path "config\calibration.json")) {
    Write-Host "Калибратор не найден, обучаю на демо-корпусе..." -ForegroundColor Yellow
    & $Python -m spanverify demo --n 240 --seed 1312 --out data\demo_dataset.jsonl
    & $Python -m spanverify calibrate --dataset data\demo_dataset.jsonl --out config\calibration.json
}

# 5. Сборка
Write-Host "Сборка PyInstaller..." -ForegroundColor Cyan
& $Python -m PyInstaller spanverify.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой." }

# 6. Комплектация release\
Write-Host "Комплектация release\..." -ForegroundColor Cyan
if (Test-Path "release") { Remove-Item "release" -Recurse -Force }
New-Item -ItemType Directory -Path "release" | Out-Null
Copy-Item "dist\spanverify.exe" "release\spanverify.exe"
Copy-Item "config" "release\config" -Recurse
Copy-Item "data" "release\data" -Recurse
Copy-Item "README.md" "release\README.md" -ErrorAction SilentlyContinue

# 7. Проверка запуска
Write-Host "Проверка собранного файла..." -ForegroundColor Cyan
& "release\spanverify.exe" selftest
if ($LASTEXITCODE -ne 0) { throw "Проверка selftest не прошла." }

$size = (Get-Item "release\spanverify.exe").Length / 1MB
Write-Host ("Готово: release\spanverify.exe ({0:N1} МБ)" -f $size) -ForegroundColor Green
Write-Host "Запуск: release\spanverify.exe            (сервис + браузер)"
Write-Host "        release\spanverify.exe analyze --file document.txt --json"
