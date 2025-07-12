import sys
sys.path.append('src')

print("Testing ConfigManager components individually...")

try:
    from src.utils.config import ModelConfig, TrainingConfig, DataConfig, KhmerVocab
    print("✓ All classes imported")
    
    print("1. Creating ModelConfig...")
    model_config = ModelConfig()
    print("✓ ModelConfig ok")
    
    print("2. Creating TrainingConfig...")
    training_config = TrainingConfig()
    print("✓ TrainingConfig ok")
    
    print("3. Creating DataConfig...")
    data_config = DataConfig()
    print("✓ DataConfig ok")
    
    print("4. Creating KhmerVocab...")
    vocab = KhmerVocab()
    print(f"✓ KhmerVocab ok, size: {len(vocab)}")
    
    print("5. Now testing ConfigManager initialization manually...")
    
    # Simulate ConfigManager.__init__ step by step
    config_path = "configs/train_config.yaml"
    
    print("6. Creating ConfigManager attributes...")
    # Simulate what ConfigManager.__init__ does
    import os
    print(f"   Config path exists: {os.path.exists(config_path)}")
    
    print("7. Creating individual components...")
    model_config2 = ModelConfig()
    training_config2 = TrainingConfig()
    data_config2 = DataConfig()
    vocab2 = KhmerVocab()
    print("✓ All components created individually")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.") 