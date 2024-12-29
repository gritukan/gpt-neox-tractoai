import os
import shutil
import sys
import time
import yt.wrapper as yt


def log(s):
    print(f"[CHECKPOINT UPLOADER] {s}", file=sys.stderr)


def get_client():
    tmp_dir = "//tmp/checkpoints_tmp"
    yt_client_config = yt.default_config.get_config_from_env()
    yt_client_config["remote_temp_files_directory"] = tmp_dir
    yt_client_config["remote_temp_tables_directory"] = tmp_dir
    yt_client = yt.YtClient(config=yt_client_config)
    yt_client.create(
        "map_node",
        tmp_dir,
        recursive=True,
        ignore_existing=True,
        attributes={
            "primary_medium": "nvme",
            "media": {
                "nvme": {
                    "replication_factor": 3,
                    "data_parts_only": False,
                },
            },
        },
    )
    return yt_client


checkpoints_path = sys.argv[1]
yt_checkpoints_path = sys.argv[2]


ytc = get_client()
log(f"Starting checkpoint uploader with checkpoint path: {checkpoints_path} and yt_path: {yt_checkpoints_path}")

while True:
    if not os.path.exists(checkpoints_path):
        log("Checkpoints dir not found, sleeping for 10 seconds")
        time.sleep(10)
        continue

    checkpoints = os.listdir(checkpoints_path)
    checkpoints = [c for c in checkpoints if c.startswith("global_step")]
    if len(checkpoints) == 0:
        log("No checkpoints found, sleeping for 10 seconds")
        time.sleep(10)
        continue
    checkpoint = checkpoints[0]
    log(f"Found checkpoint: {checkpoint}")
    checkpoint_path = f"{checkpoints_path}/{checkpoint}"
    try:
        with open(f"{checkpoint_path}/completed.txt", "r") as f:
            data = f.read()
            expected = "yay!"
            if data != expected:
                raise Exception(f"Checkpoint {checkpoint} contains different data than completed.txt: {data}")
            
        log(f"Uploading checkpoint {checkpoint}")

        def dfs(path):
            local_path = f"{checkpoints_path}/{path}"
            yt_path = f"{yt_checkpoints_path}/{path}"

            if local_path == f"{checkpoint_path}/completed.txt":
                return

            if os.path.isdir(local_path):
                log(f"Creating directory {yt_path}")
                ytc.create("map_node", yt_path, ignore_existing=True, recursive=True)
                for f in os.listdir(local_path):
                    dfs(f"{path}/{f}")
            else:
                log(f"Uploading file {yt_path}")
                with open(local_path, "rb") as f:
                    ytc.write_file(yt_path, f)
        dfs(checkpoint)

        peer_id = int(os.environ["YT_JOB_COOKIE"])
        ytc.set(f"{yt_checkpoints_path}/{checkpoint}/@peer_{peer_id}_completed", True)

        log(f"Checkpoint {checkpoint} uploaded successfully")
        log(f"Removing checkpoint {checkpoint}")
        shutil.rmtree(checkpoint_path)
    except Exception as e:
        log(f"Checkpoint isn't ready {checkpoint}: {e}, sleeping for 10 seconds")
        time.sleep(10)
        continue
