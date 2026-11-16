"""Record pre-registration audit metadata corrections without losing receipts.

No model prediction had run when these corrections were made. The selected
public cohort was byte-identical throughout; only audit explanatory metadata
changed. This script never opens the source annotation or any bbox.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mmbench_gui_new_source_v1"
AUDIT = RUN / "audit.json"
PUBLIC = RUN / "public.json"
INTERMEDIATE = RUN / "audit_superseded_lineage_v1.json"
INITIAL = RUN / "audit_superseded_initial_reconstructed.json"
RECEIPT = RUN / "audit_correction_receipt.json"
PUBLIC_SHA = "e09fb504ddf606eff10edbc7d82e1ad0a73d7f13f1b405b241a99b62ff43d649"
FINAL_LINEAGE = ("V36/V37 training images inherit the 112-image V35 TRAIN cohort; "
                 "the 112-image V35 public manifest is explicitly included in the history hash universe "
                 "alongside the earlier V32 inventory")


def canon(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        assert path.read_bytes() == content, f"Receipt changed: {path}"
    else:
        path.write_bytes(content)


def main() -> None:
    assert sha(PUBLIC) == PUBLIC_SHA
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if INTERMEDIATE.exists():
        intermediate = json.loads(INTERMEDIATE.read_text(encoding="utf-8"))
    else:
        intermediate = audit
        assert intermediate["history_lineage_note"] != FINAL_LINEAGE
        write_once(INTERMEDIATE, canon(intermediate))
    assert intermediate["public_sha256"] == PUBLIC_SHA
    assert intermediate["selected_images"] == 500 and intermediate["selected_queries"] == 1000
    initial = dict(intermediate)
    for name in ("new_source_scope", "history_lineage_note", "one_type_only_images"):
        initial.pop(name)
    write_once(INITIAL, canon(initial))
    final = dict(intermediate, history_lineage_note=FINAL_LINEAGE)
    if audit != final:
        AUDIT.write_bytes(canon(final))
    else:
        assert AUDIT.read_bytes() == canon(final)
    receipt = dict(complete=True,
                   reason="Pre-registration audit metadata correction; no sample, label, weight or prediction changed",
                   initial_audit_reconstructed_after_unrecorded_removal=True,
                   initial_audit_sha256=sha(INITIAL), intermediate_audit_sha256=sha(INTERMEDIATE),
                   final_audit_sha256=sha(AUDIT), public_sha256_unchanged=sha(PUBLIC),
                   initial_to_intermediate_fields=("new_source_scope", "history_lineage_note", "one_type_only_images"),
                   intermediate_to_final_field="history_lineage_note",
                   public_model_forward_calls=0, protocol_registered_before_correction=False)
    write_once(RECEIPT, canon(receipt))
    print(json.dumps({"final_audit_sha256": sha(AUDIT), "public_sha256": sha(PUBLIC),
                      "receipt_sha256": sha(RECEIPT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
