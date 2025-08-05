@echo off
REM Activate conda env
call E:\Installations\anaconda3\condabin\conda.bat activate LegalMedAi

REM Change to project directory
cd /d E:\GenAi_Projects\LegalClaimGPT

REM Start FastAPI in new terminal
start cmd /k "uvicorn app.api:app --reload"

REM Wait a few seconds to ensure FastAPI is up
timeout /t 5 >nul

REM Launch Streamlit app
streamlit run app/streamlit_app.py
