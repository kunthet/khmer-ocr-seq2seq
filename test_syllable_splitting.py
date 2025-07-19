#!/usr/bin/env python3
"""
Test script to validate Khmer syllable splitting functionality.
"""

import sys
from pathlib import Path

# Add src to path to import khtext modules
sys.path.append(str(Path(__file__).parent / "src"))

def test_syllable_splitting():
    """Test the syllable splitting functionality with sample texts."""
    
    try:
        from khtext.subword_cluster import split_syllables_advanced
        print("Successfully imported split_syllables_advanced")
    except ImportError as e:
        print(f"Failed to import split_syllables_advanced: {e}")
        return
    
    # Simple test texts
    test_texts = [
        "ភាគ៤៧",  # Very short: 5 chars
        "ពោជ្ឈង្គសំយុត្ត",  # Medium: 15 chars
        "ផ្ទីដីជលផលមាន ២០៤.៣៨១ ហិចតា",  # Long: 27 chars
    ]
    
    print("Testing Khmer Syllable Splitting")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        try:
            print(f"\nTest {i}: Text length: {len(text)} characters")
            print("-" * 40)
            
            # First test: Check if split_syllables_advanced works
            syllables = split_syllables_advanced(text)
            print(f"Syllables: {len(syllables)} parts")
            for j, syl in enumerate(syllables[:5]):  # Show first 5 syllables
                print(f"  Syllable {j+1}: '{syl}' (length: {len(syl)})")
            
            # Test splitting with different target lengths
            for target_length in [5, 10, 15]:
                chunks = split_text_by_syllables(text, target_length, syllables)
                print(f"Target {target_length:2d}: {len(chunks)} chunks")
                for j, chunk in enumerate(chunks):
                    print(f"  Chunk {j+1}: length {len(chunk)}")
            
        except Exception as e:
            print(f"Error processing text {i}: {e}")
            continue

def split_text_by_syllables(text, target_length, syllables=None):
    """
    Split text into chunks of approximately target_length while respecting syllable boundaries.
    
    Args:
        text (str): Text to split
        target_length (int): Target length for each chunk
        syllables (list): Pre-computed syllables (optional)
        
    Returns:
        list: List of text chunks
    """
    if len(text) <= target_length:
        return [text]
    
    # Use provided syllables or compute them
    if syllables is None:
        from khtext.subword_cluster import split_syllables_advanced
        syllables = split_syllables_advanced(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for syllable in syllables:
        syllable_length = len(syllable)
        
        # If adding this syllable would exceed target length and we have content
        if current_length + syllable_length > target_length and current_chunk:
            # Save current chunk
            chunk_text = ''.join(current_chunk)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # Start new chunk
            current_chunk = [syllable]
            current_length = syllable_length
        else:
            # Add to current chunk
            current_chunk.append(syllable)
            current_length += syllable_length
    
    # Add final chunk if it has content
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")
    
    try:
        import regex
        print("✓ regex module available")
    except ImportError:
        print("✗ regex module not available")
        return False
    
    # Check if the subword_cluster module exists
    src_path = Path(__file__).parent / "src" / "khtext" / "subword_cluster.py"
    if src_path.exists():
        print(f"✓ subword_cluster.py found at {src_path}")
    else:
        print(f"✗ subword_cluster.py not found at {src_path}")
        return False
    
    return True

if __name__ == "__main__":
    if check_dependencies():
        test_syllable_splitting()
    else:
        print("Dependencies not met. Cannot proceed with test.") 