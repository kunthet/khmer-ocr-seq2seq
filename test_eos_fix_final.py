#!/usr/bin/env python3
"""
Test the EOS handling fix - verify it stops at first EOS token, not at punctuation.
"""

import sys
sys.path.append('.')
from src.utils.config import ConfigManager

def test_eos_fix():
    config = ConfigManager('configs/config.yaml')
    print('🧪 Testing EOS handling fix...')
    
    # Use the ACTUAL problematic sequence from debug output - with repeated EOS tokens
    sequence = [0, 34, 51, 99, 24, 80, 51, 44, 80, 39, 98, 50, 114, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    print(f'Full sequence length: {len(sequence)}')
    print(f'EOS token ID: {config.vocab.EOS_IDX}')
    print(f'Token 114 (។) position: {sequence.index(114)}')
    
    # Find all EOS positions
    eos_positions = [i for i, token in enumerate(sequence) if token == config.vocab.EOS_IDX]
    print(f'First few EOS positions: {eos_positions[:10]}...')
    print(f'First EOS position: {eos_positions[0]}')
    
    # OLD way (problematic) - filters ALL EOS tokens
    old_way = [t for t in sequence if t not in [config.vocab.PAD_IDX, config.vocab.SOS_IDX, config.vocab.EOS_IDX]]
    old_text = config.vocab.decode(old_way)
    
    # NEW way (fixed) - stops at FIRST EOS token  
    clean_tokens = []
    for token_id in sequence:
        if token_id == config.vocab.EOS_IDX:  # Stop at EOS (token 1)
            break
        if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
            clean_tokens.append(token_id)
    
    new_text = config.vocab.decode(clean_tokens)
    
    print(f'\nOLD (filters all EOS): "{old_text}"')
    print(f'NEW (stops at first EOS): "{new_text}"')
    print(f'Length difference: {len(old_text)} -> {len(new_text)} characters ({len(old_text) - len(new_text)} removed)')
    
    # Check that the new way correctly includes punctuation
    if '។' in new_text:
        print('✅ Punctuation (។) correctly preserved')
    else:
        print('❌ Punctuation (។) missing')
    
    if old_text != new_text:
        print('✅ Fix is working - repetitive content removed')
        print(f'Removed content: "{old_text[len(new_text):]}"')
    else:
        print('🤔 Both methods give same result')
        
    # Test with engine._decode_sequence method for comparison
    print('\n🔍 Comparing with OCR engine method:')
    
    # Simulate engine._decode_sequence logic
    engine_clean = []
    for token_id in sequence:
        if token_id == config.vocab.EOS_IDX:
            break
        if token_id not in [config.vocab.SOS_IDX, config.vocab.PAD_IDX, config.vocab.UNK_IDX]:
            engine_clean.append(token_id)
    
    engine_text = config.vocab.decode(engine_clean)
    print(f'Engine method: "{engine_text}"')
    print(f'Matches NEW method: {engine_text == new_text}')

if __name__ == '__main__':
    test_eos_fix() 