#!/bin/bash
echo "Starting Health Data Ingestion API..."
echo "API will be available at http://10.164.42.5:8000"
echo ""
echo "To test the API, visit http://10.164.42.5:8000 in your browser"
echo ""
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
