@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo Setting up MSVC Build Tools Environment...
echo ========================================================

REM Search for vcvars64.bat in standard installation directories
set "VCVARS="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

if defined VCVARS (
    call "%VCVARS%"
) else (
    echo Note: vcvars64.bat not found in default paths. Attempting to use existing PATH compiler...
)

echo.
echo ========================================================
echo 1. Compiling Test Suite (test_all.exe)...
echo ========================================================

cl /nologo /W3 /O2 /Iinclude /Isrc ^
   /Fe:test_all.exe ^
   tests/test_all.c ^
   src/config.c ^
   src/database.c ^
   src/face_engine.c ^
   src/matcher.c ^
   src/vision_utils.c ^
   src/third_party/sqlite3.c ^
   src/third_party/cJSON.c

if %ERRORLEVEL% NEQ 0 (
    echo Compilation of test_all.exe Failed!
    exit /b 1
)

echo.
echo ========================================================
echo 2. Compiling Enrollment CLI (enroll_employee.exe)...
echo ========================================================

cl /nologo /W3 /O2 /Iinclude /Isrc ^
   /Fe:enroll_employee.exe ^
   src/enroll_employee.c ^
   src/config.c ^
   src/database.c ^
   src/face_engine.c ^
   src/matcher.c ^
   src/vision_utils.c ^
   src/third_party/sqlite3.c ^
   src/third_party/cJSON.c

echo.
echo ========================================================
echo 3. Compiling Event Responder CLI (respond_event.exe)...
echo ========================================================

cl /nologo /W3 /O2 /Iinclude /Isrc ^
   /Fe:respond_event.exe ^
   src/respond_event.c ^
   src/config.c ^
   src/database.c ^
   src/face_engine.c ^
   src/matcher.c ^
   src/vision_utils.c ^
   src/third_party/sqlite3.c ^
   src/third_party/cJSON.c

echo.
echo ========================================================
echo 4. Compiling Main CCTV Engine (run_camera.exe)...
echo ========================================================

cl /nologo /W3 /O2 /Iinclude /Isrc ^
   /Fe:run_camera.exe ^
   src/run_camera.c ^
   src/config.c ^
   src/database.c ^
   src/face_engine.c ^
   src/matcher.c ^
   src/vision_utils.c ^
   src/api_dispatcher.c ^
   src/third_party/sqlite3.c ^
   src/third_party/cJSON.c ^
   winhttp.lib

echo.
echo ========================================================
echo 5. Running Test Suite...
echo ========================================================
test_all.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo [SUCCESS] ALL BINARIES COMPILED & UNIT TESTS PASSED!
    echo ========================================================
    echo Generated executables:
    echo   - test_all.exe
    echo   - enroll_employee.exe
    echo   - respond_event.exe
    echo   - run_camera.exe
) else (
    echo.
    echo [FAILED] UNIT TESTS FAILED!
    exit /b 1
)
