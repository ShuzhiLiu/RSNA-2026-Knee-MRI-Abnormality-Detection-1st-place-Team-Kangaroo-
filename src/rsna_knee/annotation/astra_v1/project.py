"""Deterministic source-span report projection; no predictions or truth inputs.

Remove whole clauses containing synovial roots in common report languages.
Retained clauses are exact source spans with only outer list markers/whitespace
trimmed. No replacement diagnosis, removal marker/count or original numbering.
"""
import re
import unicodedata

ROOTS = ('synov', 'sinov', 'υμεν', 'συνοβ', 'синов', '滑膜', '활막')
# Split periods even without following whitespace (common in concatenated
# reports), except between two digits. All separators stay in source spans.
BOUNDARY = re.compile(r'[!?;\n]+|(?<!\d)\.|\.(?!\d)')
PREFIX = re.compile(r'^\s*(?:(?:[>*•\-]+|\d+[.)](?!\d))\s*)+')


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(c))


def project(report):
    boundaries = [0] + [m.end() for m in BOUNDARY.finditer(report)] + [len(report)]
    boundaries = sorted(set(boundaries))
    retained, removed, formatting = [], [], []
    for start, end in zip(boundaries, boundaries[1:]):
        raw = report[start:end]
        matches = [root for root in ROOTS if root in normalize(raw)]
        if matches:
            removed.append({'start': start, 'end': end, 'text': raw, 'matched_roots': matches})
            continue
        prefix = PREFIX.match(raw)
        a = start + (prefix.end() if prefix else 0)
        while a < end and report[a].isspace():
            a += 1
        b = end
        while b > a and report[b-1].isspace():
            b -= 1
        value = report[a:b]
        if not value or not any(c.isalpha() for c in value):
            formatting.append({'start': start, 'end': end, 'text': raw})
            continue
        retained.append({'start': a, 'end': b, 'text': value,
                         'source_clause_start': start, 'source_clause_end': end})
    result = '\n'.join(x['text'] for x in retained)
    assert not any(root in normalize(result) for root in ROOTS)
    assert all(report[x['start']:x['end']] == x['text'] for x in retained+removed+formatting)
    return {'projected_report': result, 'retained_spans': retained,
            'removed_spans': removed, 'formatting_only_spans': formatting}
