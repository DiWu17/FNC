"""A fresh v9 cohort, using the audited v8 historical exclusion algorithm."""
from pathlib import Path


def main():
    # Execute a namespaced copy: old sealed sources and outputs remain intact.
    source = Path('scripts/prepare_point_aligned.py').read_text(encoding='utf-8')
    source = source.replace("OUT = Path('runs/point_aligned_v8')", "OUT = Path('runs/context_verifier_v9')")
    source = source.replace('seed=20261111', 'seed=20261121').replace("'fresh_v8'", "'fresh_v9'")
    namespace = {'__name__': '_v9_preparation'}
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
