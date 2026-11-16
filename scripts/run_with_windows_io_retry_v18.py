"""I/O-only launcher for immutable V18 scripts on Windows.

Retry an atomic temporary-file replacement while a reader holds a Windows
handle. No scientific source, numerical operation, checkpoint, or policy is
changed. Fresh workers spawned by the confirmation parent inherit this launcher.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'runs/full_scaling_v18/io_repair/retry_events.jsonl'
WORKER = ROOT / 'scripts/run_mvp_confirmation_worker_v18.py'
ALLOWED = {ROOT / f'scripts/{name}' for name in [
    'run_mvp_confirmation_worker_v18.py', 'run_mvp_confirmation_v18.py',
    'run_coordinate_confirmation_v18.py', 'run_scaling_effects_v18.py']}


def install_retries():
    original = os.replace

    def replace(source, destination, *args, **kwargs):
        for attempt in range(40):
            try:
                return original(source, destination, *args, **kwargs)
            except PermissionError as error:
                eligible = (getattr(error, 'winerror', None) in (5,32,33)
                            and Path(source).name.endswith('.tmp')
                            and Path(destination).resolve().is_relative_to(ROOT)
                            and not args and not kwargs)
                if not eligible or attempt == 39:
                    raise
                try:
                    LOG.parent.mkdir(parents=True, exist_ok=True)
                    with LOG.open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),
                                                     pid=os.getpid(), destination=str(destination),
                                                     attempt=attempt+1,winerror=error.winerror))+'\n')
                except OSError:
                    pass
                time.sleep(.1)

    os.replace = replace
    return original


def install_child_launcher():
    original = subprocess.Popen

    # asyncio.windows_utils subclasses Popen during a later import. Keep the
    # class interface intact while forwarding the same child arguments.
    class LaunchPopen(original):
        def __init__(self, args, *positional, **kwargs):
            if isinstance(args,(list,tuple)):
                command=list(args)
                for i,value in enumerate(command):
                    if str(value).replace('\\','/').casefold() == str(WORKER).replace('\\','/').casefold():
                        command[i:i+1] = [str(Path(__file__).resolve()), '--script', str(WORKER)]
                        args=command
                        break
            super().__init__(args,*positional,**kwargs)

    subprocess.Popen = LaunchPopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script',required=True)
    args,remaining=parser.parse_known_args()
    script=Path(args.script).resolve()
    if script not in ALLOWED:
        raise ValueError('Launcher restricted to the registered V18 execution scripts')
    install_retries()
    install_child_launcher()
    print(json.dumps(dict(io_only_launcher=str(Path(__file__).resolve()),script=str(script),
                          atomic_replace_retry_seconds=4)),flush=True)
    sys.argv=[str(script),*remaining]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    main()
