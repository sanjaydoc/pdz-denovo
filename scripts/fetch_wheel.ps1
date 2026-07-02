# =============================================================================
# fetch_wheel.ps1  --  Resumable HTTP(S) downloader (survives slow links)
#
# Downloads a large file using HTTP Range requests, appending to a partial file
# so it can resume after interruption. Self-limits each run to a time budget so
# it returns before external timeouts; re-run to resume from where it left off.
#
# Usage:
#   powershell -File scripts\fetch_wheel.ps1 -Url <url> -OutFile <path> [-BudgetSeconds 1500]
# =============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [int]$BudgetSeconds = 1500
)

$ErrorActionPreference = "Stop"
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12

# Determine total size via HEAD.
$head = [System.Net.HttpWebRequest]::Create($Url)
$head.Method = "HEAD"
$hresp = $head.GetResponse()
$total = $hresp.ContentLength
$hresp.Close()

$existing = 0L
if (Test-Path $OutFile) { $existing = (Get-Item $OutFile).Length }

if ($existing -ge $total -and $total -gt 0) {
    Write-Host "COMPLETE: $OutFile already $([math]::Round($existing/1MB,1)) MB (>= total)."
    exit 0
}

Write-Host ("Total: {0:N1} MB | Have: {1:N1} MB | Resuming from offset {2}" -f ($total/1MB), ($existing/1MB), $existing)

$req = [System.Net.HttpWebRequest]::Create($Url)
$req.AddRange([long]$existing)
$req.Timeout = 60000
$req.ReadWriteTimeout = 60000
$resp = $req.GetResponse()
$stream = $resp.GetResponseStream()

$fs = New-Object System.IO.FileStream($OutFile, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
$buf = New-Object byte[] 1048576
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$startBytes = $existing
try {
    while ($sw.Elapsed.TotalSeconds -lt $BudgetSeconds) {
        $n = $stream.Read($buf, 0, $buf.Length)
        if ($n -le 0) { break }
        $fs.Write($buf, 0, $n)
        $existing += $n
    }
} finally {
    $fs.Close(); $stream.Close(); $resp.Close(); $sw.Stop()
}

$got = ($existing - $startBytes) / 1MB
$rate = if ($sw.Elapsed.TotalSeconds -gt 0) { $got / $sw.Elapsed.TotalSeconds } else { 0 }
Write-Host ("This run: +{0:N1} MB in {1:N0}s ({2:N2} MB/s)" -f $got, $sw.Elapsed.TotalSeconds, $rate)
Write-Host ("Progress: {0:N1} / {1:N1} MB ({2:N1}%)" -f ($existing/1MB), ($total/1MB), (100.0*$existing/$total))

if ($existing -ge $total) {
    Write-Host "COMPLETE"
    exit 0
} else {
    Write-Host "INCOMPLETE - re-run to resume"
    exit 3
}
