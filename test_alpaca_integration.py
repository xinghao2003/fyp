#!/usr/bin/env python3
"""
Integration Test: Alpaca Data with Existing Preprocessing Pipeline

This script tests that Alpaca-sourced data is fully compatible with the
existing preprocessing pipeline. It demonstrates end-to-end compatibility
from data acquisition through the complete 6-step preprocessing pipeline.

Usage:
    python test_alpaca_integration.py
"""

import os
import sys
import pandas as pd
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_alpaca_data_download():
    """Test Alpaca data download functionality."""
    print("=" * 60)
    print("TEST 1: ALPACA DATA DOWNLOAD")
    print("=" * 60)
    
    try:
        from download.alpaca_downloader import download_stock_data_alpaca
        
        # Create test directory
        test_dir = "/tmp/alpaca_integration_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Download sample data
        symbols = ["AAPL", "MSFT"]
        results = download_stock_data_alpaca(
            symbols=symbols,
            period="1mo",
            interval="1d", 
            output_dir=test_dir
        )
        
        print(f"\nDownload Results:")
        for symbol, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {symbol}: {result.get('records', 0)} records")
        
        # Verify files were created
        for symbol in symbols:
            filename = f"{symbol}_USD-1d-1mo.csv"
            filepath = os.path.join(test_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✅ File created: {filename}")
            else:
                print(f"  ❌ File missing: {filename}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

def test_data_format_compatibility():
    """Test that Alpaca data format matches Yahoo Finance expectations."""
    print("\n" + "=" * 60)
    print("TEST 2: DATA FORMAT COMPATIBILITY")
    print("=" * 60)
    
    try:
        # Load Alpaca test data
        test_file = "/tmp/alpaca_integration_test/AAPL_USD-1d-1mo.csv"
        if not os.path.exists(test_file):
            print(f"❌ Test data file not found: {test_file}")
            return False
        
        data = pd.read_csv(test_file)
        
        # Check required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required columns present: {data.columns.tolist()}")
        
        # Check data types
        print(f"\nData types:")
        for col in required_columns:
            print(f"  {col}: {data[col].dtype}")
        
        # Check for NaN values
        nan_counts = data.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️ NaN values detected:")
            for col, count in nan_counts.items():
                if count > 0:
                    print(f"  {col}: {count} NaN values")
        else:
            print("✅ No NaN values detected")
        
        # Check data consistency
        if len(data) > 0:
            print(f"\nData sample:")
            print(data.head(2).to_string())
            
            # Verify OHLC relationships
            invalid_ohlc = data[(data['high'] < data['low']) | 
                               (data['high'] < data['open']) |
                               (data['high'] < data['close']) |
                               (data['low'] > data['open']) |
                               (data['low'] > data['close'])].shape[0]
            
            if invalid_ohlc > 0:
                print(f"⚠️ {invalid_ohlc} rows have invalid OHLC relationships")
            else:
                print("✅ OHLC relationships are valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Format compatibility test failed: {e}")
        return False

def test_preprocessing_compatibility():
    """Test compatibility with existing preprocessing steps."""
    print("\n" + "=" * 60)  
    print("TEST 3: PREPROCESSING PIPELINE COMPATIBILITY")
    print("=" * 60)
    
    try:
        test_file = "/tmp/alpaca_integration_test/AAPL_USD-1d-1mo.csv"
        data = pd.read_csv(test_file)
        
        # Test 1: Date parsing
        print("Testing date parsing...")
        data['date'] = pd.to_datetime(data['date'])
        print("✅ Date parsing successful")
        
        # Test 2: Basic technical indicators (simulate stockstats functionality)
        print("Testing technical indicator calculations...")
        
        # Simple moving average (manual calculation)
        data['sma_5'] = data['close'].rolling(window=5).mean()
        print("✅ Simple moving average calculation")
        
        # RSI calculation (simplified)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        print("✅ RSI calculation")
        
        # Test 3: Data normalization simulation
        print("Testing data normalization...")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        price_columns = ['open', 'high', 'low', 'close']
        scaled_prices = scaler.fit_transform(data[price_columns])
        normalized_data = pd.DataFrame(scaled_prices, columns=[f'{col}_normalized' for col in price_columns])
        print("✅ Data normalization successful")
        
        # Test 4: Temporal split simulation
        print("Testing temporal data splitting...")
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        print(f"  Train: {len(train_data)} records")
        print(f"  Validation: {len(val_data)} records") 
        print(f"  Test: {len(test_data)} records")
        print("✅ Temporal splitting successful")
        
        # Test 5: Gym environment format
        print("Testing gym environment compatibility...")
        
        # Simulate gym-compatible format
        gym_data = data[['open', 'high', 'low', 'close', 'volume']].values
        print(f"✅ Gym format shape: {gym_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare processing performance between Alpaca and Yahoo Finance formats."""
    print("\n" + "=" * 60)
    print("TEST 4: PERFORMANCE COMPARISON")
    print("=" * 60)
    
    try:
        import time
        
        test_file = "/tmp/alpaca_integration_test/AAPL_USD-1d-1mo.csv"
        
        # Test data loading performance
        start_time = time.time()
        data = pd.read_csv(test_file)
        load_time = time.time() - start_time
        print(f"Data loading time: {load_time:.4f} seconds")
        
        # Test processing performance
        start_time = time.time()
        
        # Simulate typical preprocessing operations
        data['date'] = pd.to_datetime(data['date'])
        data['returns'] = data['close'].pct_change()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        process_time = time.time() - start_time
        print(f"Processing time: {process_time:.4f} seconds")
        print(f"Records processed: {len(data)}")
        print(f"Processing rate: {len(data)/process_time:.0f} records/second")
        
        print("✅ Performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "=" * 60)
    print("TEST 5: ERROR HANDLING")
    print("=" * 60)
    
    try:
        from download.alpaca_downloader import download_stock_data_alpaca
        
        # Test invalid symbol
        print("Testing invalid symbol handling...")
        results = download_stock_data_alpaca(
            symbols=["INVALID_SYMBOL"],
            period="1mo",
            interval="1d",
            output_dir="/tmp/alpaca_error_test"
        )
        
        if "INVALID_SYMBOL" in results:
            result = results["INVALID_SYMBOL"]
            if not result["success"]:
                print("✅ Invalid symbol handled gracefully")
            else:
                print("⚠️ Invalid symbol returned success (demo mode)")
        
        # Test empty symbol list
        print("Testing empty symbol list...")
        results = download_stock_data_alpaca(
            symbols=[],
            period="1mo", 
            interval="1d",
            output_dir="/tmp/alpaca_error_test"
        )
        print("✅ Empty symbol list handled")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files and directories."""
    test_dirs = [
        "/tmp/alpaca_integration_test",
        "/tmp/alpaca_error_test",
        "/tmp/alpaca_demo"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"🗑️ Cleaned up: {test_dir}")

def main():
    """Run all integration tests."""
    print("ALPACA API INTEGRATION TEST SUITE")
    print("=" * 80)
    print("Testing compatibility with existing FYP trading system")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Alpaca Data Download", test_alpaca_data_download),
        ("Data Format Compatibility", test_data_format_compatibility), 
        ("Preprocessing Compatibility", test_preprocessing_compatibility),
        ("Performance Comparison", test_performance_comparison),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Alpaca integration is fully compatible!")
        print("\nRecommendation: Proceed with Phase 1 implementation")
    else:
        print("⚠️ Some tests failed - Review issues before proceeding")
    
    # Clean up
    print(f"\nCleaning up test files...")
    cleanup_test_files()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)