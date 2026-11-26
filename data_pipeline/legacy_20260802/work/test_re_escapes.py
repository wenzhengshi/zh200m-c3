import re

p3 = re.compile(r'[（(]\s*[\u2018\u2019\u201c\u201d\x27\x22]{1,2}\s*[）)]')
tests = [
    "（）",
    "(\u201c\u201d)",
    "(\u2018\u2019)",
    "(\u0027\u0027)",
    "(\u0022\u0022)",
    "（\u201c\u201d）",
]
for t in tests:
    print(repr(t), bool(p3.search(t)))
