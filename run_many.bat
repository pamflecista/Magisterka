@echo off

python magisterka\Run_Number.py
set /p  number=<number.txt
set /p cmd_var=<paths.txt
set /a number2=number+2

( FOR /L %%i IN (%number%,1,%number2%) DO ( %cmd_var% %%i))



