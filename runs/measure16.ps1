Set-Location "G:\ml\Latest_Chals\work-ch2-morphometry"
$py = "C:\Users\nakul\AppData\Local\Programs\Python\Python312\python.exe"
$data = "G:\ml\Latest_Chals\challenge 2\dataset"
& $py code\train.py --data $data --out runs\m16_pre --epochs 16 --cpi 4 --val-every 3 --workers 0 --threads 6 2>&1 | Out-File -Encoding utf8 runs\m16_pre.log
& $py code\train.py --data $data --out runs\m16_rnd --epochs 16 --cpi 4 --val-every 3 --workers 0 --threads 6 --no-pretrained 2>&1 | Out-File -Encoding utf8 runs\m16_rnd.log
"MEASURE_DONE" | Out-File -Encoding utf8 runs\measure16.done
