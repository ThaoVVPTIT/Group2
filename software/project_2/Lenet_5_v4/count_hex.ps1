$files = Get-ChildItem "d:\THUC TAP\New folder\weights_hex\*.hex"
foreach ($f in $files) {
    $lines = (Get-Content $f.FullName | Where-Object { $_.Trim() -ne '' }).Count
    Write-Output ("{0,-30} {1} lines" -f $f.Name, $lines)
}
