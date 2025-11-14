"""
Wrapper script to run the dashboard from root directory
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now import and run dashboard
from dashboard import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
