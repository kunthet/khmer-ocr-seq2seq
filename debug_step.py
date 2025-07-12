import sys
sys.path.append('src')

from src.utils.config import ConfigManager

print("Creating config...")
config = ConfigManager()

print("Accessing training config...")
training_config = config.training_config

print("Checking attributes...")
print(f"Has learning_rate attr: {hasattr(training_config, 'learning_rate')}")

try:
    lr = training_config.learning_rate
    print(f"Learning rate accessed: {lr}")
    print(f"Type: {type(lr)}")
except Exception as e:
    print(f"Error accessing learning_rate: {e}")

print("Done") 