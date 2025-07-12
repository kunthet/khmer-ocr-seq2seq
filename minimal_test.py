import sys
sys.path.append('src')

print("Testing individual imports...")

try:
    print("1. Importing yaml...")
    import yaml
    print("   ✓ yaml ok")
except Exception as e:
    print(f"   ✗ yaml error: {e}")

try:
    print("2. Importing dataclass...")
    from dataclasses import dataclass
    print("   ✓ dataclass ok")
except Exception as e:
    print(f"   ✗ dataclass error: {e}")

try:
    print("3. Importing torch...")
    import torch
    print("   ✓ torch ok")
except Exception as e:
    print(f"   ✗ torch error: {e}")

try:
    print("4. Importing config classes...")
    from src.utils.config import ModelConfig, TrainingConfig, DataConfig
    print("   ✓ config classes ok")
except Exception as e:
    print(f"   ✗ config classes error: {e}")

try:
    print("5. Importing KhmerVocab...")
    from src.utils.config import KhmerVocab
    vocab = KhmerVocab()
    print(f"   ✓ KhmerVocab ok, size: {len(vocab)}")
except Exception as e:
    print(f"   ✗ KhmerVocab error: {e}")

print("All tests complete.") 