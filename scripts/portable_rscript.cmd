@echo off
set "R_HOME=%~dp0..\.r-env\Lib\R"
set "PATH=%~dp0..\.r-env\Lib\R\bin\x64;%~dp0..\.r-env\Library\bin;%PATH%"
"%~dp0..\.r-env\Lib\R\bin\Rscript.exe" %*
