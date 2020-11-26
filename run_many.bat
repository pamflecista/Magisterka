@echo off
python Run_number.py
set /p  number=<number.txt
set /p cmd_var=<paths.txt
(FOR /L %%i IN (%number%,1,3) DO (%cmd_var% %%i))



