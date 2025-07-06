#!/usr/bin/env python3
import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")

def start_server():
    """Start the FastAPI server"""
    try:
        # Install requirements first
        install_requirements()
        
        # Set environment variables
        os.environ.setdefault("PORT", "3000")
        
        # Start uvicorn server
        subprocess.call([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--host", "0.0.0.0", 
            "--port", os.environ.get("PORT", "3000")
        ])
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()