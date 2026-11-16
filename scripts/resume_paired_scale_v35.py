"""Resume sealed V35 observations; only retry transient Windows JSON renames."""
import json
import time
import run_paired_scale_v35 as experiment

original_write = experiment.write


def retry_write(path, value):
    for attempt in range(10):
        try:
            return original_write(path, value)
        except PermissionError:
            if attempt == 9:
                raise
            with (experiment.OUT / "io_retry_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at_utc": experiment.now(), "path": str(path), "attempt": attempt + 1}) + "\n")
            time.sleep(min(.05 * 2 ** attempt, 1))


if __name__ == "__main__":
    amendment = experiment.read(experiment.OUT / "resume01_amendment.json")
    assert experiment.sha(__file__) == amendment["wrapper_sha256"]
    assert experiment.sha(experiment.OUT / "protocol.json") == amendment["original_protocol_sha256"]
    for path, digest in amendment["pre_resume_archive_sha256"].items():
        assert experiment.sha(path) == digest, path
    experiment.write = retry_write
    try:
        experiment.main()
    except BaseException as error:
        experiment.status("failed", error=repr(error), traceback=experiment.traceback.format_exc())
        raise
