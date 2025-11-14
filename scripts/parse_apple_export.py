"""
Parse Apple Health Export and Integrate with Existing Data

This script:
1. Parses export.xml directly using xml.etree
2. Converts to CSV format matching our schema
3. Merges with existing API data (deduplicates)
4. Updates files/ directory with complete dataset
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_and_integrate():
    print("="*60)
    print("APPLE HEALTH EXPORT PARSER & INTEGRATOR")
    print("="*60)

    # Paths
    export_path = Path("apple_health_export/export.xml")
    output_dir = Path("files")
    output_dir.mkdir(exist_ok=True)

    if not export_path.exists():
        print(f"\n[ERROR] Export file not found: {export_path}")
        return

    # Step 1: Parse Apple Health XML
    print(f"\n[STEP 1] Parsing Apple Health export...")
    print(f"  File: {export_path}")
    print(f"  Size: {export_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  This may take several minutes...")

    # Parse XML incrementally
    records_by_type = defaultdict(list)
    workouts = []

    print(f"\n[STEP 2] Reading XML and extracting records...")

    # Use iterparse for memory efficiency
    context = ET.iterparse(str(export_path), events=('start', 'end'))
    context = iter(context)
    event, root = next(context)

    record_count = 0
    workout_count = 0

    for event, elem in context:
        if event == 'end':
            if elem.tag == 'Record':
                # Extract record data
                record = {
                    'type': elem.get('type', ''),
                    'source_name': elem.get('sourceName', ''),
                    'source_version': elem.get('sourceVersion', ''),
                    'unit': elem.get('unit', ''),
                    'creation_date': elem.get('creationDate', ''),
                    'start_date': elem.get('startDate', ''),
                    'end_date': elem.get('endDate', ''),
                    'value': elem.get('value', '')
                }

                record_type = record['type']
                records_by_type[record_type].append(record)
                record_count += 1

                if record_count % 100000 == 0:
                    print(f"  Processed {record_count:,} records...")

                # Clear element to save memory
                elem.clear()

            elif elem.tag == 'Workout':
                # Extract workout data
                workout = {
                    'workoutActivityType': elem.get('workoutActivityType', ''),
                    'duration': elem.get('duration', ''),
                    'durationUnit': elem.get('durationUnit', ''),
                    'totalDistance': elem.get('totalDistance', ''),
                    'totalDistanceUnit': elem.get('totalDistanceUnit', ''),
                    'totalEnergyBurned': elem.get('totalEnergyBurned', ''),
                    'totalEnergyBurnedUnit': elem.get('totalEnergyBurnedUnit', ''),
                    'sourceName': elem.get('sourceName', ''),
                    'startDate': elem.get('startDate', ''),
                    'endDate': elem.get('endDate', ''),
                    'creationDate': elem.get('creationDate', '')
                }
                workouts.append(workout)
                workout_count += 1

                # Clear element to save memory
                elem.clear()

        # Clear root periodically
        if event == 'end' and record_count % 100000 == 0:
            root.clear()

    print(f"  Total records extracted: {record_count:,}")
    print(f"  Total workouts extracted: {workout_count:,}")
    print(f"  Unique record types: {len(records_by_type)}")

    # Step 3: Process by type
    print(f"\n[STEP 3] Converting to CSV and merging with existing data...")

    processed_types = 0
    total_new_records = 0
    total_existing_records = 0

    for record_type, records in records_by_type.items():
        if not records:
            continue

        # Create DataFrame
        new_df = pd.DataFrame(records)

        # Output file
        output_file = output_dir / f"{record_type}.csv"

        # Merge with existing data
        if output_file.exists() and output_file.stat().st_size > 10:
            try:
                existing_df = pd.read_csv(output_file)

                # Track counts
                existing_count = len(existing_df)
                new_count = len(new_df)

                # Combine
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Remove duplicates - keep last (prefer newer data)
                dedup_columns = ['type', 'start_date', 'value']
                combined_df = combined_df.drop_duplicates(subset=dedup_columns, keep='last')

                final_count = len(combined_df)
                added_count = final_count - existing_count

                total_existing_records += existing_count
                total_new_records += added_count

                combined_df.to_csv(output_file, index=False)

                if added_count > 0:
                    print(f"  + {record_type}: Added {added_count:,} new ({existing_count:,} -> {final_count:,})")
                elif processed_types < 20:  # Only show first 20 "no change" messages
                    print(f"  = {record_type}: No new records ({existing_count:,} existing)")

            except Exception as e:
                print(f"  ! {record_type}: Error - {e}, saving new data")
                new_df.to_csv(output_file, index=False)
                total_new_records += len(new_df)
        else:
            # No existing file, create new
            new_df.to_csv(output_file, index=False)
            total_new_records += len(new_df)
            print(f"  * {record_type}: New file with {len(new_df):,} records")

        processed_types += 1

    # Step 4: Process workouts
    print(f"\n[STEP 4] Processing workouts...")
    if workouts:
        df_workouts = pd.DataFrame(workouts)
        workouts_file = output_dir / "workouts.csv"

        if workouts_file.exists() and workouts_file.stat().st_size > 10:
            existing_workouts = pd.read_csv(workouts_file)
            combined_workouts = pd.concat([existing_workouts, df_workouts], ignore_index=True)
            combined_workouts = combined_workouts.drop_duplicates(
                subset=['workoutActivityType', 'startDate', 'duration'], keep='last'
            )
            combined_workouts.to_csv(workouts_file, index=False)
            print(f"  Workouts: {len(combined_workouts):,} total")
        else:
            df_workouts.to_csv(workouts_file, index=False)
            print(f"  Workouts: Created with {len(df_workouts):,} records")
    else:
        print(f"  No workout data found")

    # Summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Metric types processed: {processed_types}")
    print(f"  Previous records: {total_existing_records:,}")
    print(f"  New records added: {total_new_records:,}")
    print(f"  Output: {output_dir.absolute()}")
    print(f"\n[NEXT STEPS]")
    print(f"  1. Restart dashboard: python dashboard.py")
    print(f"  2. For new API data: python process_bronze.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        parse_and_integrate()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Parsing stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
