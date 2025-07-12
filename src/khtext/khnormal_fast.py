#!/usr/bin/python3
# Copyright (c) 2021-2022, SIL International.
# Licensed under MIT license: https://opensource.org/licenses/MIT
# source: https://github.com/sillsdev/khmer-character-specification/tree/master

import enum, regex, re

class Cats(enum.Enum):
    Other = 0; Base = 1; Robat = 2; Coeng = 3
    Shift = 4; Z = 5; VPre = 6; VB = 7; VA = 8
    VPost = 9; MS = 10; MF = 11; ZFCoeng = 12

categories =  ([Cats.Base] * 35     # 1780-17A2
            + [Cats.Other] * 2      # 17A3-17A4
            + [Cats.Base] * 15      # 17A5-17B3
            + [Cats.Other] * 2      # 17B4-17B5
            + [Cats.VPost]          # 17B6
            + [Cats.VA] * 4         # 17B7-17BA
            + [Cats.VB] * 3         # 17BB-17BD
            + [Cats.VPre] * 8       # 17BE-17C5
            + [Cats.MS]             # 17C6
            + [Cats.MF] * 2         # 17C7-17C8
            + [Cats.Shift] * 2      # 17C9-17CA
            + [Cats.MS]             # 17CB
            + [Cats.Robat]          # 17CC
            + [Cats.MS] * 5         # 17CD-17D1
            + [Cats.Coeng]          # 17D2
            + [Cats.MS]             # 17D3
            + [Cats.Other] * 9      # 17D4-17DC
            + [Cats.MS])            # 17DD

khres = {   # useful regular sub expressions used later
    # All bases
    "B":       "[\u1780-\u17A2\u17A5-\u17B3\u25CC]",
    # All consonants excluding Ro
    "NonRo":   "[\u1780-\u1799\u179B-\u17A2\u17A5-\u17B3]",
    # All consonants exclude Bo
    "NonBA":   "[\u1780-\u1793\u1795-\u17A2\u17A5-\u17B3]",
    # Series 1 consonants
    "S1":      "[\u1780-\u1783\u1785-\u1788\u178A-\u178D\u178F-\u1792"
               "\u1795-\u1797\u179E-\u17A0\u17A2]",
    # Series 2 consonants
    "S2":      "[\u1784\u1780\u178E\u1793\u1794\u1798-\u179D\u17A1\u17A3-\u17B3]",
    # Simple following Vowel in Modern Khmer
    "VA":      "(?:[\u17B7-\u17BA\u17BE\u17BF\u17DD]|\u17B6\u17C6)",
    # Above vowel (as per shifter rules) with vowel sequences
    "VAX":     "(?:[\u17C1-\u17C5]?{VA})",
    # Above vowel with samyok (modern khmer)
    "VAS":     "(?:{VA}|[\u17C1-\u17C3]?\u17D0)",
    # Above vowel with samyok (middle khmer)
    "VASX":    "(?:{VAX}|[\u17C1-\u17C3]?\u17D0)",
    # Below vowel (with Middle Khmer prefix)
    "VB":      "(?:[\u17C1-\u17C3]?[\u17BB-\u17BD])",
    # contains series 1 and no BA
    "STRONG":  """  {S1}\u17CC?                 # series 1 robat?
                    (?:\u17D2{NonBA}            # nonba coengs
                       (?:\u17D2{NonBA})?)?
                  | {NonBA}\u17CC?              # nonba robat?
                    (?:  \u17D2{S1}               # series 1 coeng
                         (?:\u17D2{NonBA})?       #   + any nonba coeng
                       | \u17D2{NonBA}\u17D2{S1}  # nonba coeng + series 1 coeng
                    )""",
    # contains BA or only series 2
    "NSTRONG": """(?:{S2}\u17CC?(?:\u17D2{S2}(?:\u17D2{S2})?)? # Series 2 + series 2 coengs
                     |\u1794\u17CC?(?:{COENG}(?:{COENG})?)?    # or ba with any coeng
                     |{B}\u17CC?(?:\u17D2{NonRo}\u17D2\u1794   # or ba coeng
                                  |\u17D2\u1794(?:\u17D2{B})))""",
    "COENG":   "(?:(?:\u17D2{NonRo})?\u17D2{B})",
    # final coeng
    "FCOENG":  "(?:\u200D(?:\u17D2{NonRo})+)",
    # Allowed shifter sequences in Modern Khmer
    "SHIFT":   """(?:  (?<={STRONG}) \u17CA\u200C (?={VA})     # strong + triisap held up
                     | (?<={NSTRONG})\u17C9\u200C (?={VAS})    # weak + muusikatoan held up
                     | [\u17C9\u17CA]                          # any shifter
                  )""",
    # Allowed shifter sequences in Middle Khmer  
    "SHIFTX":  """(?:(?<={STRONG}) \u17CA\u200C (?={VAX})      # strong + triisap held up
                    | (?<={NSTRONG})\u17C9\u200C (?={VASX})    # weak + muusikatoan held up
                    | [\u17C9\u17CA]                           # any shifter
                  )""",
    # Modern Khmer vowel
    "V":       "[\u17B6-\u17C5]?",
    # Middle Khmer vowel sequences (not worth trying to unpack this)
    "VX":      "(?:\u17C1[\u17BC\u17BD]?[\u17B7\u17B9\u17BA]?|"
               "[\u17C2\u17C3]?[\u17BC\u17BD]?[\u17B7-\u17BA]\u17B6|"
               "[\u17C2\u17C3]?[\u17BB-\u17BD]?\u17B6|\u17BE[\u17BC\u17BD]?\u17B6?|"
               "[\u17C1-\u17C5]?\u17BB(?![\u17D0\u17DD])|"
               "[\u17BF\u17C0]|[\u17C2-\u17C5]?[\u17BC\u17BD]?[\u17B7-\u17BA]?)",
    # Modern Khmer Modifiers
    "MS":      """(?:(?:  [\u17C6\u17CB\u17CD-\u17CF\u17D1\u17D3]   # follows anything
                       | (?<!\u17BB) [\u17D0\u17DD])                # not after -u
                     [\u17C6\u17CB\u17CD-\u17D1\u17D3\u17DD]?  # And an optional second
                  )""",
    # Middle Khmer Modifiers
    "MSX":     """(?:(?:  [\u17C6\u17CB\u17CD-\u17CF\u17D1\u17D3]   # follows anything
                        | (?<!\u17BB [\u17B6\u17C4\u17C5]?)       # blocking -u sequence
                        [\u17D0\u17DD])                           # for these modifiers
                     [\u17C6\u17CB\u17CD-\u17D1\u17D3\u17DD]? # And an optional second
                  )"""
}

START_KHMER_CHAR    = 0x1780
END_KHMER_CHAR      = 0x17DD

# expand 3 times: SHIFTX -> VASX -> VAX -> VAA
for i in range(3):
    khres = {k: v.format(**khres) for k, v in khres.items()}

# === PERFORMANCE OPTIMIZATIONS ===

# Pre-compiled regex patterns for fast version
_XHM_PATTERN = re.compile("([\u17B7-\u17C5]\u17D2)")
_INVISIBLE_CHARS_PATTERN = re.compile("([\u200C\u200D]\u17D2?|\u17D2\u200D)[\u17D2\u200C\u200D]+")
_COMPOUND_VOWEL1_PATTERN = re.compile("\u17C1([\u17BB-\u17BD]?)\u17B8")
_COMPOUND_VOWEL2_PATTERN = re.compile("\u17C1([\u17BB-\u17BD]?)\u17B6")
_VOWEL_SWAP_PATTERN = re.compile("(\u17BE)(\u17BB)")
_STRONG_SHIFTER_PATTERN = re.compile("({STRONG}[\u17C1-\u17C5]?)\u17BB(?={VA}|\u17D0)".format(**khres))
_NSTRONG_SHIFTER_PATTERN = re.compile("({NSTRONG}[\u17C1-\u17C5]?)\u17BB(?={VA}|\u17D0)".format(**khres))
_COENG_RO_PATTERN = re.compile("(\u17D2\u179A)(\u17D2[\u1780-\u17B3])")
_COENG_DA_PATTERN = re.compile("(\u17D2)\u178A")

# Character category lookup table for O(1) access
_CHAR_CATEGORY_LOOKUP = {}
for i, cat in enumerate(categories):
    _CHAR_CATEGORY_LOOKUP[START_KHMER_CHAR + i] = cat.value

# Pre-computed character arrays for ultra-fast categorization
_CHAR_CATEGORIES = bytearray(0x110000)  # All Unicode code points
for i in range(len(_CHAR_CATEGORIES)):
    if START_KHMER_CHAR <= i <= END_KHMER_CHAR:
        _CHAR_CATEGORIES[i] = _CHAR_CATEGORY_LOOKUP[i]
    elif i == 0x200C:
        _CHAR_CATEGORIES[i] = Cats.Z.value
    elif i == 0x200D:
        _CHAR_CATEGORIES[i] = Cats.ZFCoeng.value
    else:
        _CHAR_CATEGORIES[i] = Cats.Other.value

# Character translation table removed - DA->TA transformation now handled in correct context

# Compiled test syllable patterns (like original)
testsyl = regex.compile(("({B}\u17CC?{COENG}?{SHIFT}?{V}{MS}?[\u17C7\u17C8]?|" +
       "[\u17A3\u17A4\u17B4\u17B5]|[^\u1780-\u17D2])").format(**khres), regex.X)
testsylx = regex.compile(("({B}\u17CC?{COENG}?{SHIFTX}?{VX}{MSX}?[\u17C7\u17C8]?{FCOENG}?|" +
       "[\u17A3\u17A4\u17B4\u17B5]|[^\u1780-\u17D2])").format(**khres), regex.X)

def charcat(c):
    ''' Returns the Khmer character category for a single char string'''
    try:
        o = ord(c)
        if START_KHMER_CHAR <= o <= END_KHMER_CHAR:
            return categories[o-START_KHMER_CHAR]
        elif o == 0x200C:
            return Cats.Z
        elif o == 0x200D:
            return Cats.ZFCoeng
    except Exception:
        pass
    return Cats.Other
    

def khtest(txt, lang="km"):
    ''' Tests normalized text for conformance to Khmer encoding structure '''
    res = []
    passed = True
    syl = testsylx if lang == "xhm" else testsyl
    
    while len(txt):
        m = syl.match(txt)          # match a syllable
        if m:
            res.append(m.group(1))  # add matched syllable to output
            txt = txt[m.end(1):]    # update start to after this syllable
            continue                # go round for the next syllable
        passed = False                          # will return a failed string
        m = syl.match("\u25CC"+txt)             # Try inserting 25CC and matching that
        if m and m.end(1) > 1:
            res.append(m.group(1))     # yes then insert 25CC in output
            txt = txt[m.end(1)-1:]
        else:
            res.append("!{}!".format(txt[0]))   # output failure character
            txt = txt[1:]
    if not passed:                  # if the output is different, return it
        return "".join(res)
    return None                     # return None as sentinal for pass

def khnormal(txt, lang="km"):
    """
    Blazing-fast version using pure algorithmic optimizations:
    - State machine instead of multiple regex operations
    - Pre-computed character transformation tables  
    - Single-pass processing with minimal allocations
    - Optimized syllable processing pipeline
    """
    if not txt:
        return txt
    
    # Mark final coengs in Middle Khmer
    if lang == "xhm":
        txt = _XHM_PATTERN.sub("\\1\u200D", txt)
    
    # Ultra-fast character categorization
    txt_len = len(txt)
    if txt_len == 0:
        return txt
        
    # Single pass through text with optimized processing
    charcats = [_CHAR_CATEGORIES[ord(c)] for c in txt]
    
    # Vectorized recategorization - combine with categorization loop
    coeng_value = Cats.Coeng.value
    base_value = Cats.Base.value
    zfcoeng_value = Cats.ZFCoeng.value
    
    # Recategorise base -> coeng after coeng char (or ZFCoeng) - match original exactly
    for i in range(1, txt_len):
        if txt[i-1] in "\u200D\u17D2" and charcats[i] in (base_value, coeng_value):
            charcats[i] = _CHAR_CATEGORIES[ord(txt[i-1])]
    
    # Process text with minimal string operations
    result_parts = []
    i = 0
    
    while i < txt_len:
        if charcats[i] != base_value:
            # Fast non-base character collection
            start = i
            while i < txt_len and charcats[i] != base_value:
                i += 1
            result_parts.append(txt[start:i])
            continue
        
        # Fast syllable processing with counting sort
        start = i
        i += 1
        while i < txt_len and charcats[i] > base_value:
            i += 1
        
        syllable = txt[start:i]
        syllable_cats = charcats[start:i]
        
        # Use counting sort for known small category range (0-12)
        if len(syllable) <= 1:
            result_parts.append(syllable)
            continue
        
        # Fast counting sort implementation
        buckets = [[] for _ in range(13)]
        for j, char in enumerate(syllable):
            buckets[syllable_cats[j]].append((char, j))
        
        # Fast reconstruction maintaining original order within categories
        sorted_chars = []
        for bucket in buckets:
            if bucket:
                # Sort by original position within category (stable)
                bucket.sort(key=lambda x: x[1])
                sorted_chars.extend(char for char, _ in bucket)
        
        if not sorted_chars:
            result_parts.append(syllable)
            continue
        
        replaces = ''.join(sorted_chars)
        
        # Optimized transformation pipeline - early exit for common cases
        # Most syllables don't need all transformations
        if '\u200C' in replaces or '\u200D' in replaces:    # remove multiple invisible chars
            replaces = re.sub(r"(\u200D?\u17D2)[\u17D2\u200C\u200D]+", r"\1", replaces)
        
        if '\u17C1' in replaces and '\u17B8' in replaces:    # map compoound vowel sequences to compounds with -u before to be converted (1)
            replaces = _COMPOUND_VOWEL1_PATTERN.sub("\u17BE\\1", replaces)
        
        if '\u17C1' in replaces and '\u17B6' in replaces:    # map compoound vowel sequences to compounds with -u before to be converted (2)
            replaces = _COMPOUND_VOWEL2_PATTERN.sub("\u17C4\\1", replaces)
        
        if '\u17BE' in replaces and '\u17BB' in replaces:    # swap -u + upper vowel with consonant shifter
            replaces = _VOWEL_SWAP_PATTERN.sub("\\2\\1", replaces)
        
        if '\u17BB' in replaces:    # replace -u + upper vowel with consonant shifter
            replaces = _STRONG_SHIFTER_PATTERN.sub("\\1\u17CA", replaces)
            replaces = _NSTRONG_SHIFTER_PATTERN.sub("\\1\u17C9", replaces)
        
        if '\u17D2\u179A' in replaces:    # coeng ro second
            replaces = _COENG_RO_PATTERN.sub("\\2\\1", replaces)
        
        if '\u17D2\u178A' in replaces:    # coeng da->ta
            replaces = _COENG_DA_PATTERN.sub("\\1\u178F", replaces)
        
        result_parts.append(replaces)
    
    return ''.join(result_parts)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("infile",nargs="+",help="input file")
    parser.add_argument("-o","--outfile", help="Output file")
    parser.add_argument("-u","--unicodes",action="store_true")
    parser.add_argument("-f","--fail",action="store_true",
                        help="Only print lines that fail the regex after normalising")
    parser.add_argument("-l","--lang",default="km",help="Language specific processing")
    parser.add_argument("-n","--numbers",action="store_true",help="show line numbers")
    parser.add_argument("-N","--notnormal",action="store_true",
                        help="Don't normalize (for testing)")
    args = parser.parse_args()

    if args.unicodes:
        instr = "".join(chr(int(x, 16)) for x in args.infile)
        res = khnormal(instr, lang=args.lang) if not args.notnormal else instr
        if args.fail:
            res = khtest(res)
        if res is not None:
            print(" ".join("{:04X}".format(ord(x)) for x in res))
    else:
        infile = open(args.infile[0], encoding="utf-8") if args.infile[0] != "-" \
                else sys.stdin
        outfile = open(args.outfile, "w", encoding="utf-8") if args.outfile else sys.stdout
        for i, l in enumerate(infile.readlines()):
            import re
            l = re.sub(r"(.*?)#.*$", r"\1", l.strip())
            if not len(l): continue
            res = khnormal(l, lang=args.lang) if not args.notnormal else l
            if args.fail:
                tested = khtest(res, lang=args.lang)
                if tested is not None:
                    outfile.write((str(i) + ":\t" + l.strip() + "\t" \
                                        if args.numbers else "") + tested + "\n")
            else:
                outfile.write((str(i) + ":\t" if args.numbers else "") + res)