# PowerShell script to set Alpaca API environment variables for current session
# Replace YOUR_API_KEY and YOUR_SECRET_KEY with your actual Alpaca credentials

$env:ALPACA_API_KEY = "PKCLL4TXCDLRN76OGRAB"
$env:ALPACA_SECRET_KEY = "ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T"

Write-Host "Environment variables set for this session." -ForegroundColor Green
Write-Host "Run 'python data.py' to test." -ForegroundColor Yellow

