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
# When the part is in place the page offers a SET button. Pressing it leaves a request beside the plan; the
# next round here measures that placement properly, builds the weld view, prints the link and stops.
#
# The board must stay in both views; the part can move freely between rounds.
param(
    [string]$Plan = "outputs/ar_fits/place/tower",
    [int]$Rounds = 0,
    [string]$Rig = "administrator@10.0.0.36",
    [string]$Welds = "outputs/welds/mainframe_ifc.json",
    [double]$Scale = 0.2,
    [switch]$Stream,                       # camera A live for the eye, camera B for the checks
    [string]$StreamCam = "52FD1B1F",
    [string]$CheckCam = "B68DE55F",
    [int]$StreamPort = 8088
)
$dev = { param($serial) "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_${serial}-video-index0" }
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$label = "place_live"
$live = "outputs/ar_captures/$label"

if ($Stream) {
    # one camera live for the operator's eye, the other doing the measuring: a shot of one camera is 5.9 s
    # against 11 s for both, and the live view has no wait at all
    $busy = Test-NetConnection -ComputerName 127.0.0.1 -Port $StreamPort -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $busy) {
        Start-Process -WindowStyle Hidden ssh -ArgumentList @("-o", "BatchMode=yes", "-f", "-N", "-L",
            "${StreamPort}:127.0.0.1:${StreamPort}", $Rig)
        Start-Sleep -Seconds 2
    }
    ssh -o BatchMode=yes $Rig "pgrep -f webcam_preview.py > /dev/null || (cd ~ && nohup python3 webcam_preview.py --devices $(& $dev $StreamCam) --port $StreamPort > /tmp/preview.log 2>&1 &)"
    Start-Sleep -Seconds 3
    Write-Host "LIVE PAGE: http://localhost:8000/$Plan/live.html    <- place against the box on this"
    Write-Host "checks run on camera $CheckCam every few seconds; the Set button lights up when it is in place"
} else {
    Write-Host "page: http://localhost:8000/$Plan/guide.html"
}
Write-Host "Ctrl+C to stop"
$n = 0
while ($true) {
    $n++
    $start = Get-Date
    $only = if ($Stream) { "--devices $(& $dev $CheckCam) --" } else { "" }
    ssh -o BatchMode=yes -o ConnectTimeout=8 $Rig "cd ~ && python3 webcam_capture.py $only shot $label > /dev/null 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "round $n`: the rig did not take the shot"
        [console]::beep(400, 600)
        Start-Sleep -Seconds 3
        continue
    }
    $want = if ($Stream) { 1 } else { 2 }
    $files = ssh -o BatchMode=yes $Rig "ls -t ~/captures/${label}_*.png 2>/dev/null | head -$want | xargs -n1 basename"
    if (-not (Test-Path $live)) { New-Item -ItemType Directory -Force $live | Out-Null }
    Remove-Item "$live/*.png" -ErrorAction SilentlyContinue
    foreach ($f in $files) { scp -o BatchMode=yes -q "${Rig}:captures/$f" "$live/" }
    ssh -o BatchMode=yes $Rig "rm -f ~/captures/${label}_*.png"

    $one = if ($Stream) { @("--view", $CheckCam) } else { @() }
    $out = docker exec -w /app cad-automation-api python tools/place_guide.py check --plan $Plan --captures $live @one 2>&1
    $line = $out | Select-String -Pattern "^(IN PLACE|ADJUST|CHECK THE PART|NOT FOUND|WRONG WAY ROUND):" | Select-Object -First 1
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
    # the operator pressed Set on the page
    $request = Join-Path $Plan "set.request"
    if (Test-Path $request) {
        Remove-Item $request -Force
        Write-Host "set: measuring this placement..."
        if ($Stream) {
            # both cameras for the measurement, so the live one has to let go of its stream first. The shot
            # re-applies and verifies the locked controls, so no re-lock is needed afterwards.
            ssh -o BatchMode=yes $Rig "pkill -f webcam_preview.py" 2>&1 | Out-Null
            Start-Sleep -Seconds 2
            ssh -o BatchMode=yes $Rig "cd ~ && python3 webcam_capture.py shot $label > /dev/null 2>&1"
            $files = ssh -o BatchMode=yes $Rig "ls -t ~/captures/${label}_*.png | head -2 | xargs -n1 basename"
            Remove-Item "$live/*.png" -ErrorAction SilentlyContinue
            foreach ($f in $files) { scp -o BatchMode=yes -q "${Rig}:captures/$f" "$live/" }
            ssh -o BatchMode=yes $Rig "rm -f ~/captures/${label}_*.png"
        }
        docker exec -w /app cad-automation-api python tools/place_guide.py set --plan $Plan --captures $live
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  not set - carry on placing"
            [console]::beep(400, 600)
        } else {
            $fit = "$Plan/fit"
            # the same calibration the placement was measured with: without it camera B is scored against camera
            # A's lens, which cost 15 points of silhouette and doubled the deviation the first time round
            $cfg = Get-Content (Join-Path $Plan "plan.json") -Raw | ConvertFrom-Json
            $extra = @()
            foreach ($c in $cfg.cam_profile) { $extra += @("--cam-profile", $c) }
            if ($cfg.stereo) { $extra += @("--stereo", $cfg.stereo) }
            if ($cfg.profile) { $extra += @("--profile", $cfg.profile) }
            docker exec -w /app cad-automation-api python tools/ar_view.py --captures $live --fit $fit `
                --welds $Welds --scale $Scale --out $fit @extra | Select-String -Pattern "welds shown|pose trust|typical|DEPARTS"
            Write-Host ""
            Write-Host "weld view: http://localhost:8000/$fit/ar_view.html"
            [console]::beep(1200, 150); [console]::beep(1600, 150); [console]::beep(2000, 150); [console]::beep(2400, 400)
            break
        }
    }
    if ($Rounds -gt 0 -and $n -ge $Rounds) { break }
}
