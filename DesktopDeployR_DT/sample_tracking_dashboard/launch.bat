taskkill /f /im Rscript.exe
taskkill /f /im chrome.exe
timeout -t 5
wscript dist\script\wsf\run.wsf