# Setting Up Alpaca API Environment Variables

## Method 1: Temporary (Current PowerShell Session)

In your PowerShell terminal, run:

```powershell
$env:ALPACA_API_KEY = "your_api_key_here"
$env:ALPACA_SECRET_KEY = "your_secret_key_here"
```

Or use the provided script:
```powershell
# Edit SET_ENV_VARS.ps1 with your credentials first, then:
. .\SET_ENV_VARS.ps1
```

**Note:** These only last for the current PowerShell session.

---

## Method 2: Permanent (User-level) via PowerShell

Run these commands in PowerShell (as Administrator for system-wide):

```powershell
[System.Environment]::SetEnvironmentVariable('ALPACA_API_KEY', 'your_api_key_here', 'User')
[System.Environment]::SetEnvironmentVariable('ALPACA_SECRET_KEY', 'your_secret_key_here', 'User')
```

**Note:** You'll need to restart your terminal/PowerShell for changes to take effect.

---

## Method 3: Permanent via Windows GUI

1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Add:
   - Variable name: `ALPACA_API_KEY`
   - Variable value: `your_api_key_here`
6. Repeat for `ALPACA_SECRET_KEY`
7. Click OK on all dialogs
8. **Restart your terminal/PowerShell**

---

## Method 4: Using .env File (Recommended for Development)

1. Create a `.env` file in the project root
2. Add your credentials:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ```
3. Add `.env` to `.gitignore` to keep credentials safe
4. Install python-dotenv: `pip install python-dotenv`
5. The code will automatically load from `.env` (if we update data.py to support it)

---

## Getting Your Alpaca API Keys

1. Sign up at https://alpaca.markets/
2. Go to your dashboard
3. Navigate to API Keys section
4. Copy your API Key ID and Secret Key

**Important:** Never commit your API keys to version control!

