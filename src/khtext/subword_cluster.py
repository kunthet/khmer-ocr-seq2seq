import regex
import unicodedata

# Fast non-regex helper functions for Unicode character checking
def is_khmer_consonant(char):
    """Check if character is a Khmer consonant (U+1780-U+17A2)"""
    return 0x1780 <= ord(char) <= 0x17A2

def is_khmer_subscript(char):
    """Check if character is Khmer coeng/subscript (U+17D2)"""
    return ord(char) == 0x17D2

def is_khmer_vowel(char):
    """Check if character is a Khmer vowel (independent or dependent)"""
    code = ord(char)
    return (0x17A5 <= code <= 0x17B3) or (0x17B6 <= code <= 0x17C8)

def is_khmer_independent_vowel(char):
    """Check if character is a Khmer independent vowel (U+17A5-U+17B3)"""
    return 0x17A5 <= ord(char) <= 0x17B3

def is_khmer_dependent_vowel(char):
    """Check if character is a Khmer dependent vowel (U+17B6-U+17C8)"""
    return 0x17B6 <= ord(char) <= 0x17C8

def is_khmer_diacritic(char):
    """Check if character is a Khmer diacritic (U+17C9-U+17D1, U+17DD)"""
    code = ord(char)
    return (0x17C9 <= code <= 0x17D1) or code == 0x17DD

def is_khmer_digit(char):
    """Check if character is a Khmer digit (U+17E0-U+17E9)"""
    return 0x17E0 <= ord(char) <= 0x17E9

def is_khmer_symbol(char):
    """Check if character is a Khmer symbol (U+17D4-U+17DC)"""
    return 0x17D4 <= ord(char) <= 0x17DC

def is_khmer_character(char):
    """Check if character is any Khmer character"""
    return 0x1780 <= ord(char) <= 0x17FF

def _classify_whitespace(text):
    """
    Classify whitespace and return appropriate tag
    
    Args:
        text (str): Whitespace text to classify
        
    Returns:
        str: Appropriate whitespace tag
    """
    if '\n' in text:
        if '\r' in text:
            return '<CRLF>'
        return '<NEWLINE>'
    elif '\t' in text:
        return '<TAB>'
    elif text == ' ':
        return '<SPACE>'
    elif len(text) > 1:
        return '<SPACES>'  # Multiple spaces
    else:
        return '<SPACE>'  # Default for other whitespace

def restore_whitespace_tags(text):
    """
    Restore whitespace tags back to actual whitespace characters
    
    Args:
        text (str): Text with whitespace tags
        
    Returns:
        str: Text with restored whitespace
    """
    replacements = {
        '<SPACE>': ' ',
        '<SPACES>': '  ',  # Two spaces as default for multiple
        '<TAB>': '\t',
        '<NEWLINE>': '\n',
        '<CRLF>': '\r\n'
    }
    
    result = text
    for tag, whitespace in replacements.items():
        result = result.replace(tag, whitespace)
    
    return result



def split_syllables(text):
    """
    Segment Khmer text into syllables using proper Khmer script patterns.
    Space tokens are replaced with special tags to preserve them during join operations.
    
    Khmer syllables typically follow this structure:
    - Initial consonant (with possible subscript consonants)
    - Optional vowels and diacritics
    - Optional final consonant
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables with space tokens as tags
        
    Example:
        >>> khmer_syllables("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        ['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ']
    """
    
    # Remove zero-width spaces for consistent results across all methods
    if not text:
        return []
    text = text.replace('\u200b', '')
    
    # Enhanced regex pattern that handles subscript consonants correctly and converts spaces to tags
    pattern = (
        r'(?:'
        # Whitespace as single tokens - will be converted to tags
        r'\s+'
        r'|'
        # Independent vowels with optional coeng + consonant combinations
        r'[\u17A5-\u17B3](?:\u17D2[\u1780-\u17A2][\u17B6-\u17D1]*)?'
        r'|'
        # Main consonant with subscript consonants and vowels - EXPANDED to handle cases like ក្ុ
        r'[\u1780-\u17A2](?:\u17D2[\u1780-\u17A2])*(?:[\u17B6-\u17D1]|\u17D2[\u17B6-\u17D1])*'
        r'|'
        # Standalone dependent vowels/diacritics
        r'[\u17B6-\u17C8\u17C9-\u17D1\u17DD]+'
        r'|'
        # Standalone subscript marker (coeng) - comes after main patterns
        r'\u17D2'
        r'|'
        # Digits (consecutive digits as single token)
        r'[\u17E0-\u17E9]+'
        r'|'
        # Symbols (each symbol as separate token)
        r'[\u17D4-\u17DC]'
        r'|'
        # Consecutive non-Khmer characters as single units
        r'[^\u1780-\u17FF\s]+'
        r')'
    )
    
    # Use single regex.findall to get all tokens including spaces
    matches = regex.findall(pattern, text)
    result = [m for m in matches if m]
    # # Filter out empty matches and convert whitespace to tags
    # result = []
    # for match in matches:
    #     if match:
    #         if match.isspace():
    #             # Convert whitespace to appropriate tag
    #             whitespace_tag = _classify_whitespace(match)
    #             result.append(whitespace_tag)
    #         else:
    #             result.append(match)
    
    return result

def split_syllables_advanced(text):
    """
    More advanced Khmer syllable segmentation using a different approach.
    This tries to better handle complex Khmer syllable structures.
    Space tokens are replaced with special tags to preserve them during join operations.
    
    Args:
        text (str): Khmer text to segment
        
    Returns:
        list: List of Khmer syllables with space tokens as tags
    """
    
    # Remove zero-width spaces for consistent results across all methods
    if not text:
        return []
    text = text.replace('\u200b', '')
    
    # Enhanced regex pattern that handles subscript consonants correctly
    pattern = (
        r'(?:'
        # Whitespace as single tokens - will be converted to tags
        r'\s+'
        r'|'
        # Independent vowel with optional coeng + consonant combination
        r'[\u17A5-\u17B3](?:\u17D2[\u1780-\u17A2][\u17B6-\u17D1]*)?'
        r'|'
        # Khmer syllable: consonant + optional subscripts + optional vowels/diacritics - EXPANDED
        r'[\u1780-\u17A2]'              # Base consonant
        r'(?:\u17D2[\u1780-\u17A2])*'   # Optional subscript consonants
        r'(?:[\u17B6-\u17D1]|\u17D2[\u17B6-\u17D1])*'  # Optional dependent vowels and diacritics, including coeng+vowel
        r'|'
        # Standalone dependent vowels/diacritics
        r'[\u17B6-\u17C8\u17C9-\u17D1\u17DD]+'
        r'|'
        # Standalone subscript marker (coeng) - comes after main patterns
        r'\u17D2'
        r'|'
        # Digits (consecutive digits as single token)
        r'[\u17E0-\u17E9]+'
        r'|'
        # Symbols (each symbol as separate token)
        r'[\u17D4-\u17DC]'
        r'|'
        # Consecutive non-Khmer characters as single units
        r'[^\u1780-\u17FF\s]+'
        r')'
    )
    
    # Use single regex.findall to get all tokens including spaces
    matches = regex.findall(pattern, text)
    result = [m for m in matches if m]
    # Filter out empty matches and convert whitespace to tags
    # result = []
    # for match in matches:
    #     if match:
    #         if match.isspace():
    #             # Convert whitespace to appropriate tag
    #             whitespace_tag = _classify_whitespace(match)
    #             result.append(whitespace_tag)
    #         else:
    #             result.append(match)
    
    return result

def segment_paragraph_to_subwords(text, method="advanced", separator="|"):
    """
    Main function to segment a Khmer paragraph into sub-words/syllables.
    
    Args:
        text (str): Input Khmer text or paragraph
        method (str): Segmentation method - "basic" or "advanced"
        separator (str): Character to use for joining syllables (default: "|")
        
    Returns:
        str: Segmented text with syllables separated by the specified separator
        
    Example:
        >>> segment_paragraph_to_subwords("អ្នកគ្រួបង្រៀនភាសាខ្មែរ")
        'អ្ន|ក|គ្រួ|ប|ង្រៀ|ន|ភា|សា|ខ្មែ|រ'
        
        >>> segment_paragraph_to_subwords("អ្នកគ្រួបង្រៀនភាសាខ្មែរ", separator="|")
        'អ្ន|ក|គ្រួ|ប|ង្រៀ|ន|ភា|សា|ខ្មែ|រ'
    """
    if method == "advanced":
        syllables = split_syllables_advanced(text)
    else:
        syllables = split_syllables(text)
    
    return separator.join(syllables)


def main():
    # Test with the sample text
    sample = "អ្នកគ្រួបង្រៀនភាសាខ្មែរ"
    
    print("=== Khmer Syllable Segmentation Demo ===")
    print(f"Original text: {sample}")
    print()
    
    # Basic usage
    print("Using the main function:")
    result = segment_paragraph_to_subwords(sample)
    print(f"Result: {result}")
    print()
    
    print("Method 2 - Advanced syllable segmentation:")
    syllables2 = split_syllables_advanced(sample)
    print("Syllables:", syllables2)
    print("result:", syllables2)
    print()
    
    # Test with a longer paragraph
    paragraph = "ខ្ញុំចង់រៀនភាសាខ្មែរ។ តើអ្នកជួយខ្ញុំបានទេ? អរគុណច្រើន!"
    print("=== Paragraph Test ===")
    print("Original:", paragraph)
    print("Segmented:", segment_paragraph_to_subwords(paragraph))

if __name__ == "__main__":
    paragraph = "ខ្ញុំចង់រៀនភាសាខ្មែរឱ្យបានល្អ ឲ្យបានល្អ។ ក្រមុំលក់នំនៅភ្នំពេញ។"
    print("=== Paragraph Test ===")
    print("Original:", paragraph)
    print("Segmented:", segment_paragraph_to_subwords(paragraph, separator="|"))
    main()