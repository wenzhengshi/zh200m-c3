s = "2012年底台湾佃农户数27,760户，在台湾农牧户总数779,375户中，比例百分之3.6"
for i, ch in enumerate(s):
    if ch in ",，、":
        print(i, repr(ch), hex(ord(ch)), "prev:", repr(s[i-1]), "next:", repr(s[i+1]))
print("all codepoints with context around commas:")
for i, ch in enumerate(s):
    if ch == "," or ch == "，" or ch == "、":
        print(f"  index={i} cp={ord(ch):04X}")
