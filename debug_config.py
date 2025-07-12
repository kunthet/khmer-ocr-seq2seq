import sys
sys.path.append('src')

from src.utils.config import ConfigManager

config = ConfigManager()
print("Training config values:")
print(f"epochs: {config.training_config.epochs}")
print(f"batch_size: {config.training_config.batch_size}")
print(f"learning_rate: {config.training_config.learning_rate}")
print(f"learning_rate type: {type(config.training_config.learning_rate)}")
print(f"1e-6: {1e-6}")
print(f"1e-6 type: {type(1e-6)}")
print(f"Equal?: {config.training_config.learning_rate == 1e-6}")
print(f"Close?: {abs(config.training_config.learning_rate - 1e-6) < 1e-10}")

print("\nDevice detection:")
print(f"Device: {config.get_device()}")

print("\nVocabulary:")
print(f"Vocab size: {len(config.vocab)}")
print(f"First few tokens: {config.vocab.vocab[:10]}") 