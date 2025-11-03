from myproject.datasets.libs.foundry_specific_utils import get_output_func
from myproject.datasets.libs.utils import (
    read_dir,
    read_dir_recursive,
    add_suffix_to_files,
    rename_files,
    get_most_recent_checkpoint
)
from myproject.datasets.libs.lib_foundry_filesystem import (
    save_foundry_files_in_dir,
    write_files_to_dataset
)

# from pyspark.sql import functions as F, types as T
from transforms.api import transform, Input, Output, configure, incremental

from datetime import datetime
import os
from pathlib import Path
import subprocess
import zipfile
import tempfile
import yaml
from ultralytics import YOLO
import torch


CLASSES = [
    "dhl_shippingLabel"
]
TS_FORMAT = '%Y-%m-%d_%H-%M-%S'
# Training Parameter
# RESUME = True  # When the transform runs incrementally this will automatically be set to True
EPOCHS = 100
BATCH = 8
IMG_SIZE = 640
WORKERS = 0
SAVE_PERIOD = 100  # save checkpoint every epoch


@incremental(
    snapshot_inputs=["train_data"],
    semantic_version=4
)
@configure(profile=['DRIVER_MEMORY_EXTRA_LARGE', "DRIVER_GPU_ENABLED", "KUBERNETES_NO_EXECUTORS"])
@transform(
    new_checkpoint=Output("ri.foundry.main.dataset.e688b10c-6e4c-40db-905d-f82c058dbd3a"),
    train_data=Input("ri.foundry.main.dataset.51d24270-eea2-43a9-8521-07f4e463814c")
)
def compute(ctx, new_checkpoint, train_data):

    output_func = get_output_func(ctx)

    if ctx.is_incremental:
        RESUME = True
        output_func("Incremental run -> Set RESUME to True.")
    else:
        RESUME = False

    # ────────────────────────────────
    # Check GPU is available.
    # ────────────────────────────────
    output_func(f"CUDA available: {torch.cuda.is_available()}")
    output_func(f"GPU count: {torch.cuda.device_count()}")
    # if nvidia-smi is present:
    try:
        output_func(subprocess.check_output(["nvidia-smi", "-L"]).decode())
    except Exception as e:
        output_func("nvidia-smi not found or no GPUs: " + str(e))

    now_str = datetime.now().strftime(TS_FORMAT)

    # ────────────────────────────────
    # Save the images (and optionally checkpoint files) from the datasets to the compute node.
    # ────────────────────────────────
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    output_func(f"Temporary directory created at {temp_dir}")

    # Get a list of the images which should be analyzed and save them as temporary files.
    _, train_temp_dir = save_foundry_files_in_dir(train_data, temp_dir, sub_dir_name="data")
    # train_temp_dir = train_temp_dir + "/data"
    output_func(f"read_dir(train_temp_dir) {read_dir(train_temp_dir)}")

    # Create folder structure           (unzip the files in the temp directory that was saved from train_data in line 79-83)
    # -tem_dir
    #   -data
    #       -images
    #           -train
    #           -val
    #           -test
    #       -labels
    #           -train
    #           -val
    #           -test
    for ff in read_dir(dir_path=train_temp_dir, file_name_ext="zip"):
        output_func(f"ff: {ff}")
        extract_dir = train_temp_dir
        filename = ff.strip(".zip").split("/")[-1]
        f1 = filename.split("_")[0]
        extract_dir = train_temp_dir + "/" + f1
        os.makedirs(extract_dir, exist_ok=True)
        f2 = filename.split("_")[1]
        extract_dir = extract_dir + "/" + f2
        os.makedirs(extract_dir, exist_ok=True)
        output_func(extract_dir)
        with zipfile.ZipFile(ff, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    output_func(f"""read_dir(train_temp_dir + "/images/train") {read_dir(train_temp_dir + "/images/train")}""")

    # ────────────────────────────────
    # Configure Parameters.                        (define and get the train and val dataset from the temp repository)
    # ────────────────────────────────
    # Paths you already have from your Foundry helper:
    ROOT_DIR = train_temp_dir
    train_imgs_dir = os.path.join(ROOT_DIR, "images", "train")
    val_imgs_dir = os.path.join(ROOT_DIR, "images", "val")

    # Where to write your YAML and checkpoints
    WORK_DIR = tempfile.mkdtemp()   # e.g. “/tmp/tmpabcd123”
    DATA_YAML = os.path.join(WORK_DIR, "data.yaml")
    MODEL_YAML = "yolo11l.yaml"
    PROJECT_DIR = os.path.join(WORK_DIR, "runs")   # Ultralytics default
    EXPER_NAME = "dhl_shippingLabel"
    output_func(f"WORK_DIR: {WORK_DIR}")
    output_func(f"DATA_YAML: {DATA_YAML}")
    output_func(f"MODEL_YAML: {MODEL_YAML}")
    output_func(f"PROJECT_DIR: {PROJECT_DIR}")
    output_func(f"EXPER_NAME: {EXPER_NAME}")

    # ────────────────────────────────
    # Save the last checkpoint if we want to resume training.         (if it is incremental: save the latest epoch and then continue from there)
    # ────────────────────────────────
    if RESUME:
        # Save the checkpoint file as a temporary file. Only take the best checkpoint
        files_list = [ii[0] for ii in list(new_checkpoint.filesystem().ls())]
        most_recent_last_checkpoint = get_most_recent_checkpoint(
            files_list, base_name="last", filename_ext="pt", ts_format=TS_FORMAT
        )
        _, checkpoint_temp_dir = save_foundry_files_in_dir(
            new_checkpoint,
            dest_dir=WORK_DIR,
            files_to_save=[most_recent_last_checkpoint],
            output_func=output_func
        )
        output_func(f"checkpoint_temp_dir: {checkpoint_temp_dir}")
        checkpoint_file = os.path.join(WORK_DIR, most_recent_last_checkpoint)

        output_func(f"""Content of WORK_DIR {read_dir(WORK_DIR)}""")
        output_func(f"""checkpoint_file {checkpoint_file}""")

    # ────────────────────────────────
    # Write the data.yaml for Ultralytics     ( for train and val dataset )
    # ────────────────────────────────
    data_dict = {
        "train": train_imgs_dir,
        "val":   val_imgs_dir,
        "nc":    len(CLASSES),
        "names": CLASSES
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(data_dict, f, sort_keys=False)
    output_func("Wrote dataset YAML to", DATA_YAML)

    # ────────────────────────────────
    # Instantiate the model.
    # ────────────────────────────────
    assert Path(DATA_YAML).exists(), f"Missing local model YAML: {DATA_YAML}"
    model = YOLO(MODEL_YAML)

    # ────────────────────────────────
    # Train or retrain model.
    # ────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("GPU requested but none found!")
    exper_name = EXPER_NAME
    if RESUME:  # Load previous model weights for retraining
        assert Path(checkpoint_file).exists(), f"Missing local checkpoint: {checkpoint_file}"
        model.load(checkpoint_file)
        exper_name += "_finetune"
    # by default, patience (early stopping) = 50 d.h stop if validation performance doesnt improve for 50 epochs
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        device=0,
        imgsz=IMG_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=exper_name,
        resume=RESUME,
        save_period=SAVE_PERIOD  # save checkpoint every epoch
    )

    output_func("Training finished. Artifacts in:", os.path.join(PROJECT_DIR, EXPER_NAME))
    output_func("Results: {str(results)}")
    with open(PROJECT_DIR + "/" + "results.txt", "w") as f_out:
        f_out.write(str(results))

    # ────────────────────────────────
    # Save checkpoint and logs to checkpoint dataset.
    # ────────────────────────────────

    output_filepaths = read_dir_recursive(PROJECT_DIR)
    # Add timestamp as suffix and rename files
    output_filepaths_w_suffix = add_suffix_to_files(output_filepaths, now_str)
    rename_files(output_filepaths, output_filepaths_w_suffix)
    output_func(f"output_filepaths: {output_filepaths}")
    output_func(f"output_filepaths_w_suffix: {output_filepaths_w_suffix}")

    # Save the new checkpoints
    write_files_to_dataset(
        dest_dataset=new_checkpoint,
        files_to_write=output_filepaths_w_suffix,
        output_func=output_func
    )
