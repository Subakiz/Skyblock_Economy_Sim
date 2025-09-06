#!/usr/bin/env python3
"""
Test script to validate the no-database mode Phase 3 CLI functionality.
This demonstrates the auction feature pipeline and ML forecasting working with NDJSON files.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, timeout=60):
    """Run a command with timeout and return success/output."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"

def test_phase3_no_db_mode():
    """Test the Phase 3 CLI in no-database mode."""
    
    print("Testing Phase 3 CLI No-Database Mode")
    print("=" * 50)
    
    # Clean up any existing data
    if Path("data").exists():
        shutil.rmtree("data")
    if Path("models").exists():
        shutil.rmtree("models")
    
    print("Step 1: Creating sample auction data...")
    success, stdout, stderr = run_command("python create_sample_auction_data.py")
    
    if not success:
        print(f"✗ Failed to create sample data: {stderr}")
        return False
    
    print(f"✓ Sample data created")
    
    # Check that data files were created
    data_files = list(Path("data").glob("**/*.ndjson"))
    if not data_files:
        print("✗ No NDJSON data files found")
        return False
    
    print(f"✓ Found {len(data_files)} data files")
    
    print("Step 2: Running Phase 3 analysis...")
    success, stdout, stderr = run_command(
        'python phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --model-type lightgbm',
        timeout=120
    )
    
    if not success:
        print(f"✗ Phase 3 analysis failed: {stderr}")
        print(f"stdout: {stdout}")
        return False
    
    print("✓ Phase 3 analysis completed successfully")
    
    # Check key outputs in the analysis results
    required_outputs = [
        "Model Training Results:",
        "Price Predictions:",
        "HYPERION:",
        "NECRON_CHESTPLATE:",
        "Market outlook:",
        "horizons trained"
    ]
    
    for output in required_outputs:
        if output not in stdout:
            print(f"✗ Missing expected output: {output}")
            return False
    
    print("✓ All expected outputs found in analysis results")
    
    # Check that auction features were created
    auction_features = Path("data/auction_features.ndjson")
    if not auction_features.exists():
        print("✗ Auction features file not created")
        return False
    
    print("✓ Auction features file created")
    
    # Check that models were saved
    model_files = list(Path("models").glob("*.pkl"))
    if len(model_files) < 6:  # Should have at least 6 models (2 items × 3 horizons)
        print(f"✗ Expected at least 6 model files, found {len(model_files)}")
        return False
    
    print(f"✓ Found {len(model_files)} model files")
    
    print("Step 3: Testing individual model training...")
    success, stdout, stderr = run_command(
        'python phase3_cli.py train HYPERION --model-type lightgbm --horizons 15 60',
        timeout=60
    )
    
    if not success:
        print(f"✗ Individual training failed: {stderr}")
        return False
    
    print("✓ Individual model training successful")
    
    print("\n" + "=" * 50)
    print("🎉 All Phase 3 no-database mode tests passed!")
    print("\nKey achievements:")
    print("- ✓ Auction feature pipeline working")
    print("- ✓ ML model training from NDJSON files")
    print("- ✓ Price predictions for auction items")
    print("- ✓ Market simulation integration")
    print("- ✓ No database dependencies")
    print("- ✓ Handles HYPERION and NECRON_CHESTPLATE")
    
    return True

def show_example_usage():
    """Show example usage of the Phase 3 CLI."""
    
    print("\n" + "=" * 50)
    print("EXAMPLE USAGE:")
    print("=" * 50)
    
    examples = [
        {
            "desc": "Analyze auction items with ML predictions and market simulation:",
            "cmd": 'python phase3_cli.py analyze "HYPERION,NECRON_CHESTPLATE" --model-type lightgbm'
        },
        {
            "desc": "Train a model for a specific item:",
            "cmd": "python phase3_cli.py train HYPERION --model-type lightgbm"
        },
        {
            "desc": "Save analysis results to file:",
            "cmd": 'python phase3_cli.py analyze "HYPERION" --output analysis_results.json'
        },
        {
            "desc": "Check Phase 3 system status:",
            "cmd": "python phase3_cli.py status"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}")
        print(f"   {example['cmd']}")

def main():
    """Main test function."""
    
    # Ensure we're in the right directory
    if not Path("phase3_cli.py").exists():
        print("Error: Must run from the repository root directory")
        return 1
    
    try:
        success = test_phase3_no_db_mode()
        
        if success:
            show_example_usage()
            return 0
        else:
            print("\n❌ Tests failed. Please check the implementation.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())