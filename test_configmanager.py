import sys
sys.path.append('src')
import os

print("Testing ConfigManager step by step...")

try:
    print("1. Importing ConfigManager...")
    from src.utils.config import ConfigManager
    print("   ✓ Import ok")
    
    print("2. Checking if config file exists...")
    config_path = "configs/train_config.yaml"
    exists = os.path.exists(config_path)
    print(f"   Config file exists: {exists}")
    
    if exists:
        print("3. Testing YAML loading...")
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        print(f"   ✓ YAML loaded, keys: {list(config_dict.keys())}")
    
    print("4. Creating ConfigManager with non-existent config...")
    config = ConfigManager(config_path="nonexistent.yaml")
    print("   ✓ ConfigManager with no file ok")
    
    print("5. Creating ConfigManager with existing config...")
    config2 = ConfigManager()  # Should use train_config.yaml
    print("   ✓ ConfigManager with file ok")
    
    print("6. Accessing learning rate...")
    lr = config2.training_config.learning_rate
    print(f"   ✓ Learning rate: {lr} (type: {type(lr)})")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.") 