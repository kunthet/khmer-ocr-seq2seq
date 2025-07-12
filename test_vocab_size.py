import sys
import os
sys.path.append('src')

from src.utils.config import KhmerVocab

v = KhmerVocab()
print(f'Final vocab size: {len(v)}')
print(f'Unique tokens: {len(set(v.vocab))}')
print(f'Has duplicates: {len(v.vocab) != len(set(v.vocab))}')

if len(v.vocab) != len(set(v.vocab)):
    # Find duplicates
    seen = set()
    duplicates = []
    for char in v.vocab:
        if char in seen:
            duplicates.append(char)
        seen.add(char)
    print(f'Duplicate characters: {duplicates}')

print('\nBreakdown:')
print(f'Special: {len(v.special_tokens)}')
print(f'Khmer nums: {len(v.khmer_numbers)}')
print(f'Arabic nums: {len(v.arabic_numbers)}')
print(f'Consonants: {len(v.consonants)}')
print(f'Indep vowels: {len(v.independent_vowels)}')
print(f'Dep vowels: {len(v.dependent_vowels)}')
print(f'Subscript: {len(v.subscript)}')
print(f'Diacritics: {len(v.diacritics)}')
print(f'Symbols: {len(v.symbols)}')

total = (len(v.special_tokens) + len(v.khmer_numbers) + len(v.arabic_numbers) + 
         len(v.consonants) + len(v.independent_vowels) + len(v.dependent_vowels) +
         len(v.subscript) + len(v.diacritics) + len(v.symbols))
print(f'Total calculated: {total}') 