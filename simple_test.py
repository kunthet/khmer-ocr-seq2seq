import sys
import os
sys.path.append('src')

try:
    print("1. Importing...")
    from src.utils.config import ConfigManager
    print("2. Creating ConfigManager...")
    config = ConfigManager()
    print("3. Checking learning rate...")
    print(f"Learning rate: {config.training_config.learning_rate}")
    print(f"Type: {type(config.training_config.learning_rate)}")
    print(f"Expected: {1e-6}")
    print(f"Equal: {config.training_config.learning_rate == 1e-6}")
    print("4. Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 