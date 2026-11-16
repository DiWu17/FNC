"""Exercise a real Windows sharing violation, separate from research processes."""
import ctypes
from ctypes import wintypes
import errno
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_with_windows_io_retry_v18 as launcher


@pytest.mark.skipif(os.name!='nt',reason='Windows file sharing semantics')
def test_replace_survives_a_reader_without_delete_sharing(tmp_path,monkeypatch):
    target=tmp_path/'status.json';target.write_text('old',encoding='utf-8')
    temp=tmp_path/'status.json.tmp';temp.write_text('new',encoding='utf-8')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                                wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(target),0x80000000,1,None,3,128,None)
    assert handle != ctypes.c_void_p(-1).value,ctypes.get_last_error()
    def release():
        time.sleep(.25)
        assert kernel.CloseHandle(handle)
    thread=threading.Thread(target=release);thread.start()
    monkeypatch.setattr(launcher,'LOG',tmp_path/'retries.jsonl')
    original=os.replace
    launcher.install_retries()
    try:
        os.replace(temp,target)
        assert target.read_text(encoding='utf-8')=='new'
        assert (tmp_path/'retries.jsonl').exists()
    finally:
        os.replace=original
        thread.join()


def test_non_windows_permission_error_is_not_suppressed(tmp_path,monkeypatch):
    calls=[]
    def denied(*args,**kwargs):
        calls.append(1)
        raise PermissionError(errno.EACCES,'permanent denial')
    monkeypatch.setattr(os,'replace',denied)
    launcher.install_retries()
    with pytest.raises(PermissionError):
        os.replace(tmp_path/'status.json.tmp',tmp_path/'status.json')
    assert calls==[1]


def test_child_launcher_preserves_late_asyncio_import():
    # A fresh interpreter reproduces the import order of the waiting analysis
    # process: install the launcher, then import asyncio through torch.
    code = '''
import sys, subprocess
sys.path.insert(0, 'scripts')
import run_with_windows_io_retry_v18 as launcher
original = subprocess.Popen
launcher.install_child_launcher()
assert issubclass(subprocess.Popen, original)
import asyncio
assert subprocess.check_output([sys.executable, '-c', 'print(17)'], text=True).strip() == '17'
print('compatible')
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='compatible'
