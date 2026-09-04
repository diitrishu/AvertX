#!/usr/bin/env python3
"""
Download publicly available industrial/occupational safety datasets
for the SIH26165 SIF precursor detection project.

Usage:
    pip install kaggle datasets pandas requests
    python download_datasets.py

For Kaggle datasets, you need a ~/.kaggle/kaggle.json API key.
Get one at: https://www.kaggle.com/settings
"""

import os
import sys
import subprocess
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
KAGGLE_DIR = DATA_DIR / "kaggle"
HUGGINGFACE_DIR = DATA_DIR / "huggingface"
PHMSA_DIR = DATA_DIR / "phmsa"
OTHER_DIR = DATA_DIR / "other"

def ensure_dirs():
    """Create all output directories."""
    for d in [KAGGLE_DIR, HUGGINGFACE_DIR, PHMSA_DIR, OTHER_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def check_kaggle():
    """Check if Kaggle API is configured."""
    try:
        import kaggle
        kaggle.api.authenticate()
        return True
    except Exception as e:
        print(f"  [SKIP] Kaggle API not configured: {e}")
        print("  To fix: pip install kaggle && place kaggle.json in ~/.kaggle/")
        return False

def download_kaggle_datasets():
    """Download datasets from Kaggle."""
    print("\n=== Kaggle Datasets ===")
    
    if not check_kaggle():
        return
    
    datasets = [
        # B1: IHM Stefanini (you already have this, but downloading raw version)
        {
            "slug": "ihmstefanini/industrial-safety-and-health-analytics-database",
            "desc": "IHM Stefanini Industrial Safety & Health Analytics (425 accidents, 12 plants, 3 countries)",
            "dir": KAGGLE_DIR / "ihm_stefanini",
        },
        # B2: Severely Injured Workers
        {
            "slug": "jboysen/injured-workers",
            "desc": "Severely Injured Workers (~22K US severe injury reports, 2015-2017)",
            "dir": KAGGLE_DIR / "injured_workers",
        },
        # B3: OSHA Accident & Injury Data
        {
            "slug": "ruqaiyaship/osha-accident-and-injury-data-1517",
            "desc": "OSHA Accident & Injury Data (construction workers, 2015-2017, with narratives)",
            "dir": KAGGLE_DIR / "osha_15_17",
        },
    ]
    
    for ds in datasets:
        print(f"\n  Downloading: {ds['desc']}")
        print(f"  URL: https://www.kaggle.com/datasets/{ds['slug']}")
        try:
            ds["dir"].mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", ds["slug"],
                "-p", str(ds["dir"]),
                "--unzip"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"  [OK] Downloaded to {ds['dir']}")
            else:
                print(f"  [FAIL] {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Download timed out after 120s")
        except Exception as e:
            print(f"  [ERROR] {e}")

def download_huggingface_datasets():
    """Download datasets from HuggingFace."""
    print("\n=== HuggingFace Datasets ===")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] 'datasets' package not installed. Run: pip install datasets")
        return
    
    datasets_to_load = [
        # D1: Injury Narrative Coding
        {
            "name": "mayerantoine/injury-narrative-coding",
            "desc": "Injury Narrative Coding (48 event codes, 7 categories, free-text narratives)",
            "dir": HUGGINGFACE_DIR / "injury_narrative_coding",
        },
    ]
    
    for ds in datasets_to_load:
        print(f"\n  Loading: {ds['desc']}")
        print(f"  URL: https://huggingface.co/datasets/{ds['name']}")
        try:
            dataset = load_dataset(ds["name"])
            ds["dir"].mkdir(parents=True, exist_ok=True)
            
            # Save each split as CSV
            for split_name, split_data in dataset.items():
                output_path = ds["dir"] / f"{split_name}.csv"
                split_data.to_csv(str(output_path))
                print(f"  [OK] Saved {split_name} split ({len(split_data)} rows) to {output_path}")
            
            # Print column info
            for split_name, split_data in dataset.items():
                print(f"  Columns ({split_name}): {split_data.column_names}")
                print(f"  Sample ({split_name}): {split_data[0]}")
                break  # Just show first split
                
        except Exception as e:
            print(f"  [ERROR] {e}")

def download_phmsa_hazmat():
    """Download PHMSA Hazmat incident reports from the Data Liberation Project."""
    print("\n=== PHMSA Hazmat Incident Reports (Data Liberation Project) ===")
    print("  URL: https://github.com/data-liberation-project/phmsa-hazmat-incident-reports")
    
    try:
        import requests
        
        # Try to download the filtered serious incidents file
        base_url = "https://raw.githubusercontent.com/data-liberation-project/phmsa-hazmat-incident-reports/main/data/processed/filtered"
        output_dir = PHMSA_DIR / "hazmat"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try the combined filtered file first
        url = f"{base_url}/serious_incidents.csv"
        print(f"  Trying: {url}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            output_path = output_dir / "serious_incidents.csv"
            output_path.write_bytes(resp.content)
            print(f"  [OK] Saved to {output_path} ({len(resp.content)} bytes)")
        else:
            print(f"  [INFO] Serious incidents file not found (status {resp.status_code})")
            print("  Falling back to cloning the repository...")
            
            # Clone the repo (shallow)
            cmd = [
                "git", "clone", "--depth", "1",
                "https://github.com/data-liberation-project/phmsa-hazmat-incident-reports.git",
                str(output_dir / "repo")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"  [OK] Cloned to {output_dir / 'repo'}")
            else:
                print(f"  [FAIL] {result.stderr.strip()}")
                
    except Exception as e:
        print(f"  [ERROR] {e}")

def download_osha_severe_injury():
    """Download OSHA Severe Injury Reports."""
    print("\n=== OSHA Severe Injury Reports ===")
    print("  URL: http://www.osha.gov/severeinjury")
    
    try:
        import requests
        
        # OSHA SIR dashboard provides CSV download
        # The exact URL may change; try the known endpoint
        url = "https://data.osha.gov/datarun/api/SevereInjury/getData?format=csv"
        output_dir = OTHER_DIR / "osha_severe_injury"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Attempting download from OSHA API...")
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (research dataset download)"
        })
        
        if resp.status_code == 200 and len(resp.content) > 1000:
            output_path = output_dir / "severe_injury_reports.csv"
            output_path.write_bytes(resp.content)
            print(f"  [OK] Saved to {output_path} ({len(resp.content)} bytes)")
        else:
            print(f"  [INFO] Direct API download returned status {resp.status_code}")
            print("  Manual download: Visit http://www.osha.gov/severeinjury and click 'Download'")
            
    except Exception as e:
        print(f"  [ERROR] {e}")

def download_msha_part50():
    """Download MSHA Part 50 accident/injury data."""
    print("\n=== MSHA Part 50 Accident/Injury Data ===")
    print("  URL: https://arlweb.msha.gov/stats/part50/p50y2k/p50y2k.htm")
    
    output_dir = OTHER_DIR / "msha_part50"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MSHA provides self-extracting files (.exe) and ZIP files
    # The exact URLs vary by year; provide guidance
    print("  MSHA Part 50 data is available as self-extracting ZIP files.")
    print("  Visit: https://arlweb.msha.gov/stats/part50/p50y2k/p50y2k.htm")
    print("  Download the most recent years' data manually.")
    print("  Also check: https://www.msha.gov/data-and-reports/reports")
    print("")
    print("  For narrative fatality reports, visit:")
    print("  https://www.msha.gov/training/training-programs-and-courses/academy-home-page/academy-library/fatality-report")

def print_summary():
    """Print summary of what was downloaded and what needs manual download."""
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    # Check what's in each directory
    for name, path in [("Kaggle", KAGGLE_DIR), ("HuggingFace", HUGGINGFACE_DIR), 
                        ("PHMSA", PHMSA_DIR), ("Other", OTHER_DIR)]:
        if path.exists():
            files = list(path.rglob("*"))
            csv_files = [f for f in files if f.suffix == ".csv"]
            print(f"\n  {name}: {len(csv_files)} CSV files found")
            for f in csv_files[:10]:
                size_kb = f.stat().st_size / 1024
                print(f"    - {f.relative_to(DATA_DIR)} ({size_kb:.1f} KB)")
            if len(csv_files) > 10:
                print(f"    ... and {len(csv_files) - 10} more")
    
    print("\n" + "-" * 70)
    print("DATASETS REQUIRING MANUAL DOWNLOAD:")
    print("-" * 70)
    print("""
  1. PHMSA Pipeline Incidents (A1)
     You already have: incident_gas_transmission_gathering_jan2010_present.txt
     Full data: https://www.phmsa.dot.gov/data-and-statistics/pipeline/source-data

  2. BSEE Offshore Incidents (A2)
     Visit: https://www.data.bsee.gov/

  3. CSB Chemical Incident Reports (A3)
     Visit: https://www.csb.gov/investigations/completed-investigations/
     Structured: https://incidents.tychodata.com/

  4. MSHA Fatality Investigation Reports (B6)
     Visit: https://www.msha.gov/data-and-reports

  5. NEISS Injury Data (B7)
     Visit: https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data

  6. UK HSE RIDDOR (B9) — narratives require FOI request
     Statistics: https://www.hse.gov.uk/statistics/sources.htm

  7. Safe Work Australia (B8) — structured only, no narratives
     Visit: https://data.safeworkaustralia.gov.au/datasets
""")

def main():
    print("=" * 70)
    print("SIH26165 — Public Safety Dataset Downloader")
    print("=" * 70)
    print(f"Output directory: {DATA_DIR}")
    
    ensure_dirs()
    
    # Download in order of priority
    download_kaggle_datasets()
    download_huggingface_datasets()
    download_phmsa_hazmat()
    download_osha_severe_injury()
    download_msha_part50()
    
    print_summary()
    
    print("\nDone! Check the README.md in this folder for the full dataset catalog.")

if __name__ == "__main__":
    main()
