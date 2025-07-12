

START_CHAR    = 0x1780
END_CHAR      = 0x17DD

CONSONANTS    = [ 0x1780 +i for i in range(35) ] 
VOWELS        = [ 0x17B6 +i for i in range(18) ]
INDEPENDENTS  = [ 0x17A3 +i for i in range(17) ]
SIGN_CHARS    = [ 0x17C8 +i for i in range(22) ]
DIGITS        = [ 0x17E0 +i for i in range(10) ]
LEK_ATTAK     = [ 0x17F0 +i for i in range(10) ]
LUNAR_DATE    = [0x19E0 + i for i in range(16)] + [0x19F0 + i for i in range(16)]  # Lunar date symbols
COMPOUND_VOWELS = ['ុំ','ាំ', 'ិះ','ុះ','េះ','ោះ', 'ឹះ', 'ើះ','ែះ','ាះ',] # ចា៎ះ (yes)

ALL_CHARS     = CONSONANTS + VOWELS + INDEPENDENTS + SIGN_CHARS + DIGITS 
ALL_CHARS_COUNT = len(ALL_CHARS) # 102



# ក ខ គ ឃ ង ច ឆ ជ ឈ ញ ដ ឋ ឌ ឍ ណ ត ថ ទ ធ ន ប ផ ព ភ ម យ រ ល វ ស ហ ឡ អ
# ឣ ឤ ឥ ឦ ឧ ឨ ឩ ឪ ឫ ឬ ឭ ឮ ឯ ឰ ឱ ឲ ឳ ឴ ឵
# ា ិ ី ឹ ឺ ុ ូ ួ ើ ៀ េ ែ ៃ ោ ៅ ំ ះ
# ៈ ៉ ៊ ់ ៌ ៍ ៎ ៏ ័ ៑ ្ ៓ ។ ៕ ៖ ៗ ៘ ៙ ៚ ៛ ៜ ៝
# ៞ ៟ ០ ១ ២ ៣ ៤ ៥ ៦ ៧ ៨ ៩ ៪ ៫ ៬ ៭ ៮ ៯
# ៰ ៱ ៲ ៳ ៴ ៵ ៶ ៷ ៸ ៹

CONSONANT_KA    = 0x1780  # ក
CONSONANT_KHA   = 0x1781  # ខ
CONSONANT_KO    = 0x1782  # គ
CONSONANT_KHO   = 0x1783  # ឃ
CONSONANT_NGO   = 0x1784  # ង
CONSONANT_CHA   = 0x1785  # ច
CONSONANT_CHHA  = 0x1786  # ឆ
CONSONANT_CHO   = 0x1787  # ជ
CONSONANT_CHHO  = 0x1788  # ឈ
CONSONANT_NHO   = 0x1789  # ញ
CONSONANT_DA    = 0x178A  # ដ
CONSONANT_DHA   = 0x178B  # ឋ
CONSONANT_DO    = 0x178C  # ឌ
CONSONANT_DHO   = 0x178D  # ឍ
CONSONANT_NA    = 0x178E  # ណ
CONSONANT_TA    = 0x178F  # ត
CONSONANT_THA   = 0x1790  # ថ
CONSONANT_TO    = 0x1791  # ទ
CONSONANT_THO   = 0x1792  # ធ
CONSONANT_NO    = 0x1793  # ន
CONSONANT_BO    = 0x1794  # ប
CONSONANT_PHA   = 0x1795  # ផ
CONSONANT_PHO   = 0x1796  # ព
CONSONANT_PHO   = 0x1797  # ភ
CONSONANT_MO    = 0x1798  # ម
CONSONANT_YO    = 0x1799  # យ
CONSONANT_RO    = 0x179A  # រ
CONSONANT_LO    = 0x179B  # ល
CONSONANT_VO    = 0x179C  # វ
CONSONANT_SSO   = 0x179D  # ឝ   # DEPRECATED
CONSONANT_SHO   = 0x179E  # ឞ   # DEPRECATED
CONSONANT_SA    = 0x179F  # ស
CONSONANT_HA    = 0x17A0  # ហ
CONSONANT_LA    = 0x17A1  # ឡ
CONSONANT_A     = 0x17A2  # អ

INDEPENDENT_A       = 0x17A3  # ឣ   # DEPRECATED
INDEPENDENT_AA      = 0x17A4  # ឤ
INDEPENDENT_I       = 0x17A5  # ឥ
INDEPENDENT_II      = 0x17A6  # ឦ
INDEPENDENT_U       = 0x17A7  # ឧ
INDEPENDENT_UK      = 0x17A8  # ឨ
INDEPENDENT_UU      = 0x17A9  # ឩ
INDEPENDENT_UUV     = 0x17AA  # ឪ
INDEPENDENT_RYU     = 0x17AB  # ឫ
INDEPENDENT_RYUU    = 0x17AC  # ឬ
INDEPENDENT_LYU     = 0x17AD  # ឭ
INDEPENDENT_LYUU    = 0x17AE  # ឮ
INDEPENDENT_E       = 0x17AF  # ឯ
INDEPENDENT_AI      = 0x17B0  # ឰ
INDEPENDENT_EI      = 0x17B1  # ឱ
INDEPENDENT_OI      = 0x17B2  # ឲ
INDEPENDENT_OUI     = 0x17B3  # ឳ
INDEPENDENT_KIV_AQ  = 0x17B4  # ឴?   
INDEPENDENT_KIV_AA  = 0x17B5  # ឵?

VOWEL_AA    = 0x17B6  # ា
VOWEL_I     = 0x17B7  # ិ
VOWEL_II    = 0x17B8  # ី
VOWEL_OE    = 0x17B9  # ឹ
VOWEL_OUE   = 0x17BA  # ឺ
VOWEL_U     = 0x17BB  # ុ
VOWEL_UU    = 0x17BC  # ូ
VOWEL_UA    = 0x17BD  # ួ
VOWEL_OE    = 0x17BE  # ើ
VOWEL_YOUE  = 0x17BF  # ឿ   # DEPRECATED
VOWEL_YEU   = 0x17C0  # ៀ
VOWEL_E     = 0x17C1  # េ
VOWEL_AE    = 0x17C2  # ែ
VOWEL_AI    = 0x17C3  # ៃ
VOWEL_AO    = 0x17C4  # ោ
VOWEL_AU    = 0x17C5  # ៅ
VOWEL_OM    = 0x17C6  # ំ
VOWEL_AH    = 0x17C7  # ះ

SIGN_YUUKALEAPINTU  = 0x17C8  # ៈ
SIGN_MUUSIKATOAN    = 0x17C9  # ៉
SIGN_TRIISAP        = 0x17CA  # ៊
SIGN_BANTOC         = 0x17CB  # ់
SIGN_ROBAT          = 0x17CC  # ៌
SIGN_TOANDAKHIAT    = 0x17CD  # ៍
SIGN_KAKABAT        = 0x17CE  # ៎
SIGN_AHSDA          = 0x17CF  # ៏
SIGN_SAMYOK         = 0x17D0  # ័
SIGN_VIRIAM         = 0x17D1  # ៑
SIGN_COENG          = 0x17D2  # ្
SIGN_BATHAMASAT     = 0x17D3  # ៓
SIGN_KHAN           = 0x17D4  # ។
SIGN_KHAN_BARIYOOSAN= 0x17D5  # ៕
SIGN_CAMNUC_PII_KUUH= 0x17D6  # ៖
SIGN_LEK_TOO        = 0x17D7  # ៗ
SIGN_LAK            = 0x17D8  # ៘
SIGN_PHNAEK_MUAN    = 0x17D9  # ៙
SIGN_KOOMUUT        = 0x17DA  # ៚
SIGN_RIEL           = 0x17DB  # ៛
SIGN_AVAKRAHA       = 0x17DC  # ៜ
SIGN_ATTHACAN       = 0x17DD  # ៝

RESERVED_CHAR_0     = 0x17DE  # ៞
RESERVED_CHAR_1     = 0x17DF  # ៟

DIGIT_0             = 0x17E0  # ០
DIGIT_1             = 0x17E1  # ១
DIGIT_2             = 0x17E2  # ២
DIGIT_3             = 0x17E3  # ៣
DIGIT_4             = 0x17E4  # ៤
DIGIT_5             = 0x17E5  # ៥
DIGIT_6             = 0x17E6  # ៦
DIGIT_7             = 0x17E7  # ៧
DIGIT_8             = 0x17E8  # ៨
DIGIT_9             = 0x17E9  # ៩

RESERVED_CHAR_2    = 0x17EA  # ៪
RESERVED_CHAR_3    = 0x17EB  # ៫
RESERVED_CHAR_4    = 0x17EC  # ៬
RESERVED_CHAR_5    = 0x17ED  # ៭
RESERVED_CHAR_6    = 0x17EE  # ៮
RESERVED_CHAR_7    = 0x17EF  # ៯
LEK_ATTAK_0        = 0x17F0  # ៰
LEK_ATTAK_1        = 0x17F1  # ៱
LEK_ATTAK_2        = 0x17F2  # ៲
LEK_ATTAK_3        = 0x17F3  # ៳
LEK_ATTAK_4        = 0x17F4  # ៴
LEK_ATTAK_5        = 0x17F5  # ៵
LEK_ATTAK_6        = 0x17F6  # ៶
LEK_ATTAK_7        = 0x17F7  # ៷
LEK_ATTAK_8        = 0x17F8  # ៸
LEK_ATTAK_9        = 0x17F9  # ៹

#  lunar date symbols is U+19E0–U+19FF:
#           0	1	2	3	4	5	6	7	 8	9	A	B	C	D	E	F
# U+19Ex	᧠	᧡	᧢	᧣	᧤	᧥	᧦	᧧	᧨	᧩	᧪	᧫	᧬	᧭	᧮	᧯
# U+19Fx	᧰	᧱	᧲	᧳	᧴	᧵	᧶	᧷	᧸	᧹	᧺	᧻	᧼	᧽	᧾	᧿



COMPLEXITY_LEVELS = {
    'SIMPLE': DIGITS + [CONSONANT_KA, CONSONANT_NGO],  # Basic shapes
    'MODERATE': CONSONANTS[:20],  # Standard consonants
    'COMPLEX': INDEPENDENTS + SIGN_CHARS,  # Complex shapes/stacking
    'COMPOUND': COMPOUND_VOWELS,  # Multi-character combinations
    'LUNAR_DATE': LUNAR_DATE,  # Lunar date symbols
}