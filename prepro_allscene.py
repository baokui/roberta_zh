import os
import random
def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False
min_len = 6
max_len = 30
path_root = 'data_allScene_pretrain/raw'
files = os.listdir(path_root)
files = [os.path.join(path_root,file) for file in files if 'all' in file]
random.shuffle(files)
for i in range(len(files)):
    with open(files[i],'r',encoding='utf-8') as f:
        s = f.read().strip().split('\n')[1:]
    s = [ss.split('\t') for ss in s]
    S = []
    for j in range(len(s)):
        t = s[j][0]
        if len(t)<min_len:
            continue
        while len(t)>0 and not _is_chinese_char(t[0]):
            t = t[1:]
        if len(t)<min_len:
            continue
        while len(t)>0 and not _is_chinese_char(t[-1]):
            t = t[:-1]
        if len(t)<min_len and len(t)>max_len:
            continue
        S.append(t)
    with open(files[i].replace('raw','raw-washed'),'w',encoding='utf-8') as f:
        f.write('\n'.join(S))
    print(i, files[i],len(S))