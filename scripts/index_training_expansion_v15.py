"""Index atomically completed screenshots while the separate downloader runs."""
import concurrent.futures
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from normalize_training_expansion_v15 import DATA,RUN,read,write,identify,local_image_path


def main():
    cases=read(RUN/'selection.json')['cases']
    finished={r['key'] for r in cases if (DATA/'image_index'/f"{r['key']}.json").exists()}
    errors=[];started=time.monotonic();last=-1
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            available=[r for r in cases if r['key'] not in finished and local_image_path(DATA/'raw'/r['dataset'],r['relative_image']).is_file()]
            futures={pool.submit(identify,row):row for row in available}
            for future in concurrent.futures.as_completed(futures):
                row=futures[future]
                try:future.result()
                except Exception as error:errors.append(dict(key=row['key'],image=row['relative_image'],error=str(error)))
                finished.add(row['key'])
                if len(finished)//100!=last:
                    last=len(finished)//100
                    print(json.dumps(dict(stage='streaming_rgb_index',indexed=len(finished),n=len(cases),errors=len(errors),seconds=round(time.monotonic()-started,1))),flush=True)
            complete=RUN/'download_complete.json'
            if complete.exists():
                downloaded=read(complete)
                if len(finished)>=downloaded['n']:
                    break
            time.sleep(5)
    write(RUN/'streaming_index_complete.json',dict(complete=downloaded['complete'] and not errors,indexed=len(finished),errors=errors))
    print(json.dumps(dict(stage='streaming_index_complete',indexed=len(finished),errors=errors)),flush=True)


if __name__=='__main__':main()
