@echo off
REM Run this once before inference.py to load your keys
REM Usage: load_env.bat

set GROQ_API_KEY_1=gsk_paste_your_key_here
set GROQ_API_KEY_2=gsk_paste_second_key_here_or_leave_as_is
set MODEL_NAME=llama-3.3-70b-versatile
set API_BASE_URL=http://localhost:7860

echo Keys loaded. Now run: python inference.py --task all
