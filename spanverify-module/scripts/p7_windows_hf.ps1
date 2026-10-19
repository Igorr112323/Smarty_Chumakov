param([string]$ReleaseDir = "release", [string]$OutputDir = "release/hf-autonomy")
$ErrorActionPreference = 'Stop'
$release = (Resolve-Path $ReleaseDir).Path
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$out = (Resolve-Path $OutputDir).Path
$layout = Join-Path $env:RUNNER_TEMP 'p7-hf-isolated'
New-Item -ItemType Directory -Force $layout | Out-Null
$pythonRoot = Split-Path (Get-Command python).Source
Copy-Item $pythonRoot (Join-Path $layout 'python') -Recurse -Force
$python = Join-Path $layout 'python/python.exe'
$libs = Join-Path $layout 'libs'
$model = Join-Path $layout 'model'
& $python -m pip install --target $libs 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'external torch preparation failed' }
& $python -m pip install --target $libs --upgrade 'transformers==5.20.1' 'sentencepiece==0.2.2' 'numpy==2.4.6'
if ($LASTEXITCODE -ne 0) { throw 'external HF libraries preparation failed' }
$env:PYTHONPATH = $libs
$env:PYTHONNOUSERSITE = '1'
$env:P7_MODEL = $model
& $python -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download('ai-forever/rugpt3small_based_on_gpt2',local_dir=os.environ['P7_MODEL'],allow_patterns=['config.json','tokenizer*','vocab*','merges.txt','special_tokens_map.json','pytorch_model.bin','*.safetensors','generation_config.json']))"
if ($LASTEXITCODE -ne 0) { throw 'external frozen model download failed' }
Copy-Item (Join-Path $release 'spanverify.exe') $layout
Copy-Item (Join-Path $release 'spanverify.pyz') $layout
Copy-Item (Join-Path $release 'frozen-p5.json') $layout
Copy-Item (Join-Path $release 'p5-runtime.json') $layout
$exe = Join-Path $layout 'spanverify.exe'
$weights = Join-Path $layout 'frozen-p5.json'
$pyz = Join-Path $layout 'spanverify.pyz'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HOME = Join-Path $layout 'offline-cache'
$env:TORCH_HOME = Join-Path $layout 'offline-torch'
$env:SPANVERIFY_HF_MODEL = $model
$env:SPANVERIFY_HF_MAX_TOKENS = '512'
$env:HTTP_PROXY = 'http://127.0.0.1:9'
$env:HTTPS_PROXY = 'http://127.0.0.1:9'
$env:NO_PROXY = '127.0.0.1,localhost'
$rules = @()
function RegistryState {
  $s = @{}
  foreach ($key in @('HKCU:\Software\SpanVerify', 'HKLM:\Software\SpanVerify')) {
    $s[$key] = if (Test-Path $key) { (Get-ItemProperty $key | ConvertTo-Json -Compress) } else { $null }
  }
  return ($s | ConvertTo-Json -Compress)
}
$before = RegistryState
$receipts = @()
try {
  foreach ($program in @($exe, $python)) {
    $name = 'P7-no-network-' + [guid]::NewGuid().ToString()
    New-NetFirewallRule -DisplayName $name -Direction Outbound -Program $program -Action Block -Profile Any | Out-Null
    $rules += $name
    if (-not (Get-NetFirewallRule -DisplayName $name).Enabled) { throw 'offline firewall rule not enabled' }
  }
  $port = 18765
  foreach ($kind in @('exe','pyz')) {
    $stdout = Join-Path $out "$kind-stdout.log"
    $stderr = Join-Path $out "$kind-stderr.log"
    if ($kind -eq 'exe') {
      $command = $exe
      $arguments = @('--mode','hf','--model',$model,'--weights',$weights,'--runtime-dir',$libs,'--port',"$port",'--no-browser')
    } else {
      $command = $python
      $arguments = @($pyz,'server','--mode','hf','--weights',$weights,'--port',"$port",'--no-browser','--quiet')
    }
    $proc = Start-Process -FilePath $command -ArgumentList $arguments -WorkingDirectory $layout -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    try {
      $ready = $false
      for ($i=0; $i -lt 120; $i++) {
        if ($proc.HasExited) { throw "$kind exited $($proc.ExitCode): $(Get-Content $stderr -Raw)" }
        try {
          $health = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3
          if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 1
      }
      if (-not $ready) { throw "$kind did not start" }
      $answerText = 'Срок хранения первичных документов составляет десять лет.'
      $body = @{answer='Срок хранения первичных документов составляет десять лет.';context='Регламент: первичные документы хранятся пять лет, вторичные документы — десять лет.';with_tokens=$true} | ConvertTo-Json -Compress
      $body | Set-Content -Encoding utf8 (Join-Path $out "$kind-request.json")
      $response = Invoke-RestMethod "http://127.0.0.1:$port/v1/verify" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json; charset=utf-8' -TimeoutSec 120
      if ($response.mode -ne 'hf' -or $response.stats.features.backend -ne 'hf') { throw 'HF surrogate fallback forbidden' }
      if ($response.spans.Count -eq 0) { throw 'mandatory attribution span missing' }
      foreach ($s in $response.spans) {
        if ($s.start -lt 0 -or $s.end -gt $answerText.Length -or $s.end -le $s.start -or $s.risk -lt 0 -or $s.risk -gt 1) { throw 'invalid span' }
      }
      $response | ConvertTo-Json -Depth 15 | Set-Content -Encoding utf8 (Join-Path $out "$kind-response.json")
      $receipts += @{kind=$kind; mode='hf'; backend='hf'; HTTP=200; offline=$true; firewall_block=$true; working_directory=$layout; model=$model; libraries=$libs; interpreter=$command; response_file="$kind-response.json"}
    } finally {
      if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    }
    $port++
  }
  if ($before -ne (RegistryState)) { throw 'application registry keys changed' }
  $hashes = @(Get-ChildItem $model -File -Recurse | Where-Object { $_.FullName -notmatch '\.cache' } | ForEach-Object { @{file=$_.FullName.Substring($model.Length+1);bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()} })
  $report = @{status='success';code=$env:GITHUB_SHA;run=$env:GITHUB_RUN_ID;offline=$true;registry_product_keys_unchanged=$true;layout=@{python='python/python.exe';libs='libs/';weights='model/';exe='spanverify.exe';pyz='spanverify.pyz';quality_bundle='frozen-p5.json';runtime='p5-runtime.json'};libs_bytes=(Get-ChildItem $libs -File -Recurse | Measure-Object Length -Sum).Sum;model_files=$hashes;receipts=$receipts}
  $report | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 (Join-Path $out 'windows-hf-autonomy.json')
  Write-Host ('::notice title=P7 Windows HF autonomy::'+(@{status='success';code=$env:GITHUB_SHA;offline=$true;exe_HF=$true;pyz_HF=$true;registry_product_keys_unchanged=$true;libs_bytes=$report.libs_bytes} | ConvertTo-Json -Compress))
} finally {
  foreach ($name in $rules) { Remove-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue }
}
