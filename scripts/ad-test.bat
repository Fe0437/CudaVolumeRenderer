@echo off
echo rendering test of the ad scene using naiveSK, regenerationSK, streamingSK, sortingSK on 3 trials and 100 iterations
pause
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mitsubaxml\ad\volpath_bsdf.xml --kernel naiveSK --scene-type MitsubaXml --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mitsubaxml\ad\volpath_bsdf.xml --kernel regenerationSK --scene-type MitsubaXml --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mitsubaxml\ad\volpath_bsdf.xml --kernel streamingSK --scene-type MitsubaXml --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mitsubaxml\ad\volpath_bsdf.xml --kernel sortingSK --scene-type MitsubaXml --interactive false --trials 3 --iterations 100
pause