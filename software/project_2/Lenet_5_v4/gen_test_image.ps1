# Script tao anh test 28x28 (784 pixel, uint8 mem)
# Background: 00 (0, se duoc RTL chuyen thanh -128)
# Foreground: FF (255, se duoc RTL chuyen thanh 127)
# Pattern: chu so "1" don gian

$lines = @()
for ($r = 0; $r -lt 28; $r++) {
    for ($c = 0; $c -lt 28; $c++) {
        $val = 0x00  # background

        # Than chu so "1": cot 14, hang 5-22
        if ($r -ge 5 -and $r -le 22 -and $c -eq 14) { $val = 0xFF }

        # Duong cheo tren (net gach cua so 1): hang 5-7, cot 12-13
        if ($r -ge 5 -and $r -le 7 -and $c -eq (13 - ($r - 5))) { $val = 0xFF }
        if ($r -ge 5 -and $r -le 6 -and $c -eq 13) { $val = 0xFF }

        # Chan de (base): hang 22, cot 12-16
        if ($r -eq 22 -and $c -ge 12 -and $c -le 16) { $val = 0xFF }

        $lines += "{0:x2}" -f $val
    }
}
$lines | Set-Content "d:\THUC TAP\New folder\test_image.mem" -Encoding ASCII
Write-Output "Created test_image.mem with $($lines.Count) entries"
