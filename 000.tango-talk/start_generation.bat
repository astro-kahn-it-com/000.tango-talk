@echo off
set PYTHON=D:\arte\work\astro-kahn-it-com\000.chatterbox-talk\python_embeded\python.exe
%PYTHON% -m pip install git+https://github.com/declare-lab/TangoFlux.git --no-deps
%PYTHON% generate.py
pause
