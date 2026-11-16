"""Conservative, model-output-independent instruction checks for grounding."""
from __future__ import annotations

import hashlib
import re
import unicodedata

PLACEHOLDERS = frozenset(('\ufffc', '\ufffd'))


def unusable_instruction_reason(text):
    if not isinstance(text, str):
        return 'instruction_is_not_text'
    visible = ''.join(c for c in text if not c.isspace() and unicodedata.category(c) not in ('Cc', 'Cf'))
    if not visible:
        return 'instruction_has_no_visible_content'
    if all(c in PLACEHOLDERS for c in visible):
        return 'instruction_contains_only_unreadable_placeholders'
    return None


def wording_tag(text):
    # These are review strata, never actionability labels or exclusion rules.
    if re.search(r'\bhover\b|\bmove (?:your |the )?(?:mouse |cursor|pointer)|\bpass your mouse|\bpoint (?:at|to)\b', text, re.I):
        return 'pointing_or_hover_wording'
    if len(text.strip()) <= 3:
        return 'short_label'
    return 'other_grounding_description'


def audit_sample(records, per_source=30):
    selected = []
    for source in sorted({r['dataset'] for r in records}):
        cohort = [r for r in records if r['dataset'] == source]
        ordered = sorted(cohort, key=lambda r: hashlib.sha256(('v15_qc1_manual90\0'+r['sample_id']).encode()).hexdigest())
        chosen = []; pixels = set(); group_counts = {}
        def take(available, limit):
            for _ in range(limit):
                choices = [r for r in available if r['pixel_sha256'] not in pixels]
                if not choices: break
                rank = {r['sample_id']:i for i,r in enumerate(ordered)}
                item = min(choices, key=lambda r:(group_counts.get(r['source_group'],0), rank[r['sample_id']]))
                chosen.append(item);pixels.add(item['pixel_sha256'])
                group_counts[item['source_group']] = group_counts.get(item['source_group'],0)+1
        take([r for r in ordered if wording_tag(r['instruction']) == 'pointing_or_hover_wording'],10)
        take([r for r in ordered if wording_tag(r['instruction']) == 'short_label'],10)
        take([r for r in ordered if any(a.get('visible_intersection_clipped') for a in r['source_annotations'])],5)
        take(ordered,per_source-len(chosen))
        if len(chosen) != per_source:
            raise ValueError('Not enough unique screenshots for manual review: '+source)
        selected.extend(dict(r,review_wording_tag=wording_tag(r['instruction']),review_status='pending') for r in chosen)
    return selected
