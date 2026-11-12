# Optional Windows Word rendering for visual QA. Requires desktop Word installed.
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot 'qa') | Out-Null
$ReportInput = Join-Path $PSScriptRoot 'report\CNN大作业报告.docx'
$ReportPdf = Join-Path $PSScriptRoot 'qa\report.pdf'
$ReportWord = New-Object -ComObject Word.Application
$ReportWord.Visible = $false
$ReportWord.DisplayAlerts = 0
try {
    $ReportDocument = $ReportWord.Documents.Open($ReportInput,$false,$true)
    $ReportDocument.ExportAsFixedFormat($ReportPdf,17)
    $ReportDocument.Close(0)
    Write-Output $ReportPdf
} finally {
    $ReportWord.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ReportWord) | Out-Null
}
