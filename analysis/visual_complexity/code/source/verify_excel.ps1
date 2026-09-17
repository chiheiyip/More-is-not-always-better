param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference = 'Stop'
$result = Get-Content -LiteralPath (Join-Path $RunDirectory 'results.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$bookPath = Join-Path $RunDirectory '六场景视觉复杂度.xlsx'
$excel = $null
$book = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.AskToUpdateLinks = $false
    $book = $excel.Workbooks.Open($bookPath, 0, $true)
    $excel.CalculateFullRebuild()
    $scores = $book.Worksheets.Item('标准化与综合分')
    $pairs = $book.Worksheets.Item('原始指标与差值')
    $raw = $book.Worksheets.Item('逐面结果与说明')
    $maxDifference = 0.0
    $keys = @('FC','DL','z_FC','z_DL','Score')
    for ($i=0; $i -lt 6; $i++) {
        for ($j=0; $j -lt 5; $j++) {
            $actual = $scores.Cells.Item(6+$i,4+$j).Value2
            if ($actual -isnot [double]) { throw "Non-numeric score at row $i column $j : $actual" }
            $expected = [double]$result.scenes[$i].($keys[$j])
            $diff = [Math]::Abs($actual-$expected)
            $maxDifference = [Math]::Max($maxDifference,$diff)
            if ($diff -gt 1e-8) { throw "Excel score mismatch: $actual versus $expected" }
        }
    }
    $keys = @('FC_C0','FC_C1','delta_FC','DL_C0','DL_C1','delta_DL')
    for ($i=0; $i -lt 3; $i++) {
        for ($j=0; $j -lt 6; $j++) {
            $actual = $pairs.Cells.Item(6+$i,2+$j).Value2
            if ($actual -isnot [double]) { throw "Non-numeric pair value at row $i column $j : $actual" }
            $expected = [double]$result.pairs[$i].($keys[$j])
            $diff = [Math]::Abs($actual-$expected)
            $maxDifference = [Math]::Max($maxDifference,$diff)
            if ($diff -gt 1e-8) { throw "Excel pair mismatch: $actual versus $expected" }
        }
    }
    $original = $raw.Range('E2').Value2
    $raw.Range('E2').Value2 = [double]$original + 0.06
    $excel.CalculateFullRebuild()
    if ([Math]::Abs($scores.Range('D6').Value2 - ([double]$result.scenes[0].FC+0.01)) -gt 1e-8) { throw 'Excel input-change recalculation failed' }
    if ([Math]::Abs($pairs.Range('D6').Value2 - ([double]$result.pairs[0].delta_FC-0.01)) -gt 1e-8) { throw 'Excel downstream recalculation failed' }
    $raw.Range('E2').Value2 = $original
    $excel.CalculateFullRebuild()
    $verification = [pscustomobject]@{engine='Microsoft Excel';version=$excel.Version;checked_numeric_results=48;maximum_absolute_difference=$maxDifference;input_change_recalculation='passed';source_workbook_opened_readonly=$true}
    $verification | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDirectory 'verification\native_excel_validation.json') -Encoding UTF8
    $verification | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $book) { $book.Close($false) }
    if ($null -ne $excel) { $excel.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
