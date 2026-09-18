# Placement guide loop (bd 6et) - PowerShell version of tools/place_loop.sh, for running from a normal
# PowerShell prompt on the laptop while the operator is at the rig.
#
#   .\tools\place_loop.ps1                        # until Ctrl+C, using the tower plan
#   .\tools\place_loop.ps1 -Rounds 1              # one check
#   .\tools\place_loop.ps1 -Plan outputs/ar_fits/place/tower_flipped
#
# Each round: both rig cameras take a full-resolution shot (about 11 s - they open one at a time, which is
# what keeps the frames uncompressed), the pair is copied here, tools/place_guide.py checks it inside the
# running API container, and the laptop beeps: three rising notes for IN PLACE, one low note otherwise.
# Leave the page open - it refreshes itself.
#
# The board must stay in both views; the part can move freely between rounds.
param(
    [string]$Plan = "outputs/ar_fits/place/tower",
    [int]$Rounds = 0,
    [string]$Rig = "administrator@10.0.0.36"
)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$label = "place_live"
$live = "outputs/ar_captures/$label"

Write-Host "page: http://localhost:8000/$Plan/guide.html"
Write-Host "Ctrl+C to stop"
$n = 0
while ($true) {
    $n++
    $start = Get-Date
    ssh -o BatchMode=yes -o ConnectTimeout=8 $Rig "cd ~ && python3 webcam_capture.py shot $label > /dev/null 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "round $n`: the rig did not take the shot"
        [console]::beep(400, 600)
        Start-Sleep -Seconds 3
        continue
    }
    $files = ssh -o BatchMode=yes $Rig "ls -t ~/captures/${label}_*.png 2>/dev/null | head -2 | xargs -n1 basename"
    if (-not (Test-Path $live)) { New-Item -ItemType Directory -Force $live | Out-Null }
    Remove-Item "$live/*.png" -ErrorAction SilentlyContinue
    foreach ($f in $files) { scp -o BatchMode=yes -q "${Rig}:captures/$f" "$live/" }
    ssh -o BatchMode=yes $Rig "rm -f ~/captures/${label}_*.png"

    $out = docker exec -w /app cad-automation-api python tools/place_guide.py check --plan $Plan --captures $live 2>&1
    $line = $out | Select-String -Pattern "^(IN PLACE|ADJUST|NOT FOUND|WRONG WAY ROUND):" | Select-Object -First 1
    $secs = [int]((Get-Date) - $start).TotalSeconds
    if ($line) {
        Write-Host "round $n ($secs s): $line"
        if ("$line" -like "IN PLACE*") {
            [console]::beep(1200, 150); [console]::beep(1600, 150); [console]::beep(2000, 300)
        } else {
            [console]::beep(700, 250)
        }
    } else {
        Write-Host "round $n ($secs s): check failed"
        $out | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" }
        [console]::beep(400, 600)
    }
    if ($Rounds -gt 0 -and $n -ge $Rounds) { break }
}
