$hexFiles = Get-ChildItem "d:\THUC TAP\New folder\weights_hex\*.hex"
foreach ($file in $hexFiles) {
    $newName = $file.Name -replace '\.hex$', '.mem'
    Rename-Item $file.FullName -NewName $newName
}

if (Test-Path "d:\THUC TAP\New folder\test_image.hex") {
    Rename-Item "d:\THUC TAP\New folder\test_image.hex" -NewName "test_image.mem"
}
Write-Output "Renamed all .hex files to .mem"
