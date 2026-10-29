# DOCX -> PDF via Microsoft Word COM (PowerShell interop).
#
# Why: the Python env has no pywin32, and installing packages is not allowed in this
# repo. PowerShell ships COM interop natively, so Word can still be driven without
# adding anything. Word renders CJK punctuation at line breaks correctly; LibreOffice
# drops a few trailing marks, which corrupts the submitted PDF text layer.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File docx_to_pdf_word.ps1 `
#       -DocxPath "C:\path\paper.docx" -PdfPath "C:\path\paper.pdf"
param(
    [Parameter(Mandatory = $true)][string]$DocxPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)

$ErrorActionPreference = 'Stop'

$docx = (Resolve-Path -LiteralPath $DocxPath).Path

# Word COM requires absolute paths; a relative output path raises "目录名无效".
if ([System.IO.Path]::IsPathRooted($PdfPath)) {
    $pdfFull = [System.IO.Path]::GetFullPath($PdfPath)
}
else {
    $pdfFull = [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $PdfPath))
}
$pdfDir = Split-Path -Parent $pdfFull
if (-not (Test-Path -LiteralPath $pdfDir)) {
    New-Item -ItemType Directory -Path $pdfDir -Force | Out-Null
}
$pdf = Join-Path $pdfDir ([System.IO.Path]::GetFileName($pdfFull))
if (Test-Path -LiteralPath $pdf) {
    Remove-Item -LiteralPath $pdf -Force
}

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    # Open(FileName, ConfirmConversions, ReadOnly, AddToRecentFiles)
    $doc = $word.Documents.Open($docx)
    # ExportAsFixedFormat(OutputFileName, ExportFormat); wdExportFormatPDF = 17.
    # SaveAs2(..., 17) raises a COMException on this machine, so use the export API.
    $doc.ExportAsFixedFormat([string]$pdf, 17)
    $doc.Close(0)
    $doc = $null
    Write-Output "word-com-ps"
}
finally {
    if ($null -ne $doc) { try { $doc.Close(0) } catch { } }
    if ($null -ne $word) { try { $word.Quit() } catch { } }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}

if (-not (Test-Path -LiteralPath $pdf)) {
    throw "Word did not produce $pdf"
}
