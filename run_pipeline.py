import os
import subprocess
import sys

def run_step(command, description):
    print(f"\n{'='*20} {description} {'='*20}")
    try:
        # Using subprocess to run scripts in separate processes to avoid namespace pollution
        result = subprocess.run(command, shell=True, check=True)
        if result.returncode != 0:
            print(f"Error: {description} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running {description}: {e}")
        sys.exit(1)

def main():
    print("Starting Full Data Pipeline...")
    
    # 1. Clean Fundamentals
    run_step(f"{sys.executable} clean_fundamentals.py", "Cleaning Fundamentals")
    
    # 2. Run ETL (Fundamentals, Technicals, Ratings)
    # Note: src/fallback_etl.py is in src, need to handle python path or run from root
    run_step(f"{sys.executable} src/fallback_etl.py", "Running ETL & Ratings")
    
    # 3. Run ML
    run_step(f"{sys.executable} src/fallback_ml.py", "Training ML Models & Predicting")
    
    print("\n" + "="*50)
    print("Pipeline Completed Successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
