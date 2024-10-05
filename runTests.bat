::
:: @file      runTests.bat
::
:: @author    David Bayer \n
::            Faculty of Information Technology \n
::            Brno University of Technology \n
::            ibayer@fit.vutbr.cz
::
:: @brief     PCG Assignment 1
::
:: @version   2024
::
:: @date      04 October   2023, 09:00 (created) \n
::

@echo off

if "%1" == "" (
  echo "Usage: <path to nbody binary>"
  exit /b -1
)

set PYTHON_VENV_DIR=PyVirtEnv

set TESTS_DIR=Tests
set TESTS_INPUT_DIR=%TESTS_DIR%\Inputs
set TESTS_OUTPUT_DIR=%TESTS_DIR%\Outputs
set TESTS_REF_OUTPUT_DIR=%TESTS_DIR%\ReferenceOutputs

if not exist %PYTHON_VENV_DIR%\Scripts\activate (
  python3 -m venv %PYTHON_VENV_DIR%
)

call %PYTHON_VENV_DIR%\Scripts\activate

python3 -m pip install -q h5py 2> nul

:: clean files
rmdir /s /q %TESTS_OUTPUT_DIR%
mkdir %TESTS_OUTPUT_DIR%
 
:: Test: Two particles on circle
echo "Two particles on circular trajectory"
.\%1 2 0.00001f 543847 128 0 32 32 %TESTS_INPUT_DIR%\circle.h5 %TESTS_OUTPUT_DIR%\circle.h5 > nul
python3 %TESTS_DIR%\test-difference.py %TESTS_OUTPUT_DIR%\circle.h5 %TESTS_REF_OUTPUT_DIR%\circle-ref.h5

:: Test:
echo "Points on line without collision"
.\%1 32 0.001f 10000 128 0 32 32 %TESTS_INPUT_DIR%\two-lines.h5 %TESTS_OUTPUT_DIR%\two-lines.h5 > nul
python3 %TESTS_DIR%/test-difference.py %TESTS_OUTPUT_DIR%\two-lines.h5 %TESTS_REF_OUTPUT_DIR%\two-lines-ref.h5


:: Test:
echo "Points on line with one collision"
.\%1 32 0.001f 45000 128 0 32 32 %TESTS_INPUT_DIR%\two-lines.h5 %TESTS_OUTPUT_DIR%\two-lines-one.h5 > nul
python3 %TESTS_DIR%\test-difference.py %TESTS_OUTPUT_DIR%\two-lines-one.h5 %TESTS_REF_OUTPUT_DIR%\two-lines-collided-45k.h5


:: Test:
echo "Points on line with several collision"
.\%1 32 0.001f 50000 128 0 32 32 %TESTS_INPUT_DIR%\two-lines.h5 %TESTS_OUTPUT_DIR%\two-lines-several.h5 > nul
python3 %TESTS_DIR%\test-difference.py %TESTS_OUTPUT_DIR%\two-lines-several.h5 %TESTS_REF_OUTPUT_DIR%\two-lines-collided-50k.h5


:: Test:
echo "Symetry globe test"
.\%1 932 0.1f 1 128 0 32 32 %TESTS_INPUT_DIR%\thompson_points_932.h5 %TESTS_OUTPUT_DIR%\thompson.h5 > nul
python3 %TESTS_DIR%\test-thompson.py %TESTS_OUTPUT_DIR%\thompson.h5 %TESTS_REF_OUTPUT_DIR%\thompson_points_932.h5 


:: Test:
echo "Stability globe test"
.\%1 932 0.00001f 15000 128 0 32 32 %TESTS_INPUT_DIR%\thompson_points_932.h5 %TESTS_OUTPUT_DIR%\thompson.h5 > nul
python3 %TESTS_DIR%\test-thompson.py %TESTS_OUTPUT_DIR%\thompson.h5 %TESTS_REF_OUTPUT_DIR%\thompson_points_932.h5

deactivate

exit /b 0
