"""Fetch the public TAG backbone mirror, still verifying official HF hashes."""
from pathlib import Path
import download_advantage_models_v17 as download

ROOT = Path(__file__).resolve().parents[1]
STAGING = 'MiniCPM-Llama3-V-2_5_modelscope_staging'


def main():
    download.MODELS = {'tag': ('openbmb/MiniCPM-Llama3-V-2_5', STAGING)}
    original_get = download.requests.get

    def mirror_get(url, **kwargs):
        filename = url.split('/')[-1].split('?')[0]
        return original_get(f'https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5/resolve/master/{filename}', **kwargs)

    download.requests.get = mirror_get
    download.main()
    staging = (ROOT / 'models' / STAGING).resolve()
    target = (ROOT / 'models/MiniCPM-Llama3-V-2_5').resolve()
    workspace_models = (ROOT / 'models').resolve()
    assert staging.parent == target.parent == workspace_models
    if not target.exists():
        staging.rename(target)
        print('Published verified TAG mirror at ' + str(target), flush=True)
    else:
        print('TAG target already exists; verified mirror retained separately at ' + str(staging), flush=True)


if __name__ == '__main__':
    main()
