$count = (Get-Content "d:\THUC TAP\New folder\test_image.mem" | Where-Object { $_.Trim() -ne '' }).Count
Write-Output "Lines: $count"
$first5 = Get-Content "d:\THUC TAP\New folder\test_image.mem" -First 5
Write-Output "First 5 lines:"
$first5 | ForEach-Object { Write-Output "  $_" }
