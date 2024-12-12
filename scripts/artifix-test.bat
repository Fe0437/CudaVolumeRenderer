@echo off
echo rendering test of the artifix scene using naiveSK, regenerationSK, streamingSK, sortingSK on 3 trials with 100 iterations
pause
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mhd\artifix_small.mhd --kernel naiveSK --scene-type VtkMha --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mhd\artifix_small.mhd --kernel regenerationSK --scene-type VtkMha --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mhd\artifix_small.mhd --kernel streamingSK --scene-type VtkMha --interactive false --trials 3 --iterations 100
"..\implementation\testing-version-executable\VolumeRenderer.exe" ..\implementation\data\mhd\artifix_small.mhd --kernel sortingSK --scene-type VtkMha --interactive false --trials 3 --iterations 100
pause