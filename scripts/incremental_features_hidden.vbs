' incremental_features.ps1 을 창 없이 실행하는 래퍼.
'
' 작업 스케줄러가 powershell.exe 를 직접 띄우면 5분마다 콘솔 창이 깜빡인다.
' WScript.Shell.Run 의 세 번째 인자 0 = 창 숨김, False = 종료를 기다리지 않음.
'
' 등록:
'   schtasks /Create /TN "scheduler-incremental-features" ^
'     /TR "wscript.exe \"F:\dev\scheduler\scripts\incremental_features_hidden.vbs\"" ^
'     /SC MINUTE /MO 5 /F
'
' 주의: 이건 어디까지나 로컬 PC 임시 운영용이다. PC 를 끄면 피처 생성이 멈추고
' 60분 뒤 추천이 0건이 된다. 정상 운영은 서버에서 돌리는 것 (docs/배포_체크리스트.md).

Dim shell, scriptDir
Set shell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & _
          scriptDir & "\incremental_features.ps1""", 0, False
