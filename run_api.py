"""
Wrapper script to run the REST API server from root directory
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now import and run API
from app import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
