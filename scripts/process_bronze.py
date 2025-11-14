"""
Process Bronze Layer Data

This script reads JSON files from the bronze layer (automated data ingestion)
and converts them to CSV format compatible with the dashboard.
"""

import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os


def process_bronze_to_csv(bronze_dir: str = "bronze", output_dir: str = "files"):
    """
    Process all JSON files from bronze layer and convert to CSV

    Args:
        bronze_dir: Directory containing bronze layer data
        output_dir: Directory to save CSV files
    """
    bronze_path = Path(bronze_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not bronze_path.exists():
        print(f"Bronze directory {bronze_dir} does not exist")
        return

    # Dictionary to accumulate records by type
    records_by_type: Dict[str, List[Dict]] = {}

    # Walk through all bronze folders
    for type_dir in bronze_path.glob("type=*"):
        hk_type = type_dir.name.replace("type=", "")

        for date_dir in type_dir.glob("date=*"):
            # Process all JSON files in this date directory
            for json_file in date_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract relevant fields
                    record = {
                        'type': hk_type,
                        'creation_date': data.get('received_ts'),
                        'start_date': data.get('start') or data.get('startDate') or data.get('dateTime'),
                        'end_date': data.get('end') or data.get('endDate'),
                        'value': data.get('value') or data.get('qty'),
                        'unit': data.get('unit'),
                        'source_name': data.get('source') or data.get('sourceName') or 'iOS App',
                        'source_version': data.get('sourceVersion') or 'unknown',
                    }

                    # Add to appropriate list
                    if hk_type not in records_by_type:
                        records_by_type[hk_type] = []

                    records_by_type[hk_type].append(record)

                except Exception as e:
                    print(f"Error processing {json_file}: {e}")

    # Convert each type to CSV
    for hk_type, records in records_by_type.items():
        if not records:
            continue

        df = pd.DataFrame(records)

        # Save to CSV
        output_file = output_path / f"{hk_type}.csv"

        # If file exists, append to it (avoiding duplicates)
        if output_file.exists() and output_file.stat().st_size > 10:
            try:
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)

                # Remove duplicates based on start_date, value, and type
                df = df.drop_duplicates(subset=['type', 'start_date', 'value'], keep='last')
            except pd.errors.EmptyDataError:
                # File exists but is empty, just use new data
                pass

        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

    print(f"\nProcessed {len(records_by_type)} health metric types")


if __name__ == "__main__":
    print("Processing bronze layer data...")
    process_bronze_to_csv()
    print("Done!")
