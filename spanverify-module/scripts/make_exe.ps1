# Сборка spanverify.exe (Windows, PyInstaller).
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_exe.ps1
#
# В GitHub Actions этот же шаг выполняет workflow build-and-release-exe.yml,
# после чего собранный файл проверяется смоук-тестом до публикации релиза.
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

Write-Host "Установка сборочных зависимостей…"
python -m pip install --upgrade pip
python -m pip install pyinstaller

Write-Host "Сборка одного файла…"
python -m PyInstaller spanverify.spec --noconfirm --clean

if (-not (Test-Path "dist\spanverify.exe")) { throw "dist\spanverify.exe не собран" }

Write-Host "Смоук-тест собранного файла…"
$process = Start-Process -FilePath "dist\spanverify.exe" -ArgumentList "--port","8765","--no-browser" -PassThru
try {
    $ready = $false
    foreach ($i in 1..40) {
        Start-Sleep -Milliseconds 750
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 3
            if ($health.status -eq "ok") { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "сервис не поднялся на порту 8765" }

    $body = @{
        answer  = "Срок хранения первичных документов составляет 3 года."
        context = "Регламент 343: срок хранения первичных документов составляет 10 лет."
    } | ConvertTo-Json
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/verify" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
    Write-Host ("Ответ сервиса: verdict={0} score={1} фрагментов={2}" -f $result.verdict, $result.score, $result.spans.Count)
    if ($null -eq $result.score -or $result.spans.Count -lt 1) { throw "смоук-тест не нашёл фрагмент" }
} finally {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "Готово: dist\spanverify.exe"
