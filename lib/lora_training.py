"""
lora-training: K12 上で LoRA 学習を実行するライブラリ。

データセット準備・設定生成・k3s ジョブ投入・デプロイを標準化。
K12 操作は全て subprocess (scp, ssh, kubectl) 経由。
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K12_HOST = "k12"
K12_DATASET_BASE = "/home/kento/lora-training/datasets"
K12_CONFIG_DIR = "/home/kento/lora-training/configs"
K12_OUTPUT_DIR = "/home/kento/lora-training/output"
K12_LORA_DIR = "/data/comfyui/models/loras"
K12_CHECKPOINT = "/data/comfyui/models/checkpoints/NoobAI-XL-Vpred-v1.0.safetensors"
DOCKER_IMAGE = "localhost/lora-training:v1"
NAMESPACE = "lora-training"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, printing it for visibility."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_dataset(
    name: str,
    image_dir: str,
    trigger_word: str,
    caption_tags: str | None = None,
    repeat: int = 10,
) -> int:
    """Copy images to K12 and generate caption .txt files.

    sd-scripts requires ``train_data_dir`` to be a **parent** directory
    containing subfolders named ``{repeat}_{concept}/``.  This function
    creates the subfolder ``{repeat}_{trigger_word}/`` inside the dataset
    directory and places images + captions there.

    Remote structure created::

        /home/kento/lora-training/datasets/{name}/
            {repeat}_{trigger_word}/
                image1.png
                image1.txt
                ...

    Args:
        name: Dataset name (used as directory name on K12).
        image_dir: Local directory containing training images.
        trigger_word: Trigger word for the LoRA.
        caption_tags: Optional comma-separated tags appended after trigger_word.
        repeat: Number of repeats (encoded in subfolder name, default 10).

    Returns:
        Number of images prepared.
    """
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect image files
    images = sorted(
        p
        for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[prepare_dataset] Found {len(images)} images in {image_dir}")

    # Build caption text
    caption = trigger_word
    if caption_tags:
        caption = f"{trigger_word}, {caption_tags}"

    # sd-scripts subfolder name: {repeat}_{trigger_word}
    subset_folder = f"{repeat}_{trigger_word}"

    # Create caption files locally in a temp dir, then scp everything
    with tempfile.TemporaryDirectory(prefix="lora_ds_") as tmpdir:
        # Mirror the subfolder structure locally
        local_subset = os.path.join(tmpdir, subset_folder)
        os.makedirs(local_subset)

        for img in images:
            # Copy image to local subset dir
            dst_img = os.path.join(local_subset, img.name)
            os.link(img, dst_img) if os.stat(img).st_dev == os.stat(tmpdir).st_dev else __import__("shutil").copy2(str(img), dst_img)

            # Create matching .txt caption
            txt_name = img.stem + ".txt"
            with open(os.path.join(local_subset, txt_name), "w") as f:
                f.write(caption)

        # Ensure remote parent directory exists
        remote_dir = f"{K12_DATASET_BASE}/{name}"
        print(f"[prepare_dataset] Creating remote directory: {remote_dir}/{subset_folder}/")
        _run(["ssh", K12_HOST, "mkdir", "-p", f"{remote_dir}/{subset_folder}"])

        # scp the subset folder contents
        print(f"[prepare_dataset] Uploading {len(images)} images + captions to {K12_HOST}:{remote_dir}/{subset_folder}/")
        _run(["scp", "-r"] + glob.glob(os.path.join(local_subset, "*")) + [f"{K12_HOST}:{remote_dir}/{subset_folder}/"])

    print(f"[prepare_dataset] Done. {len(images)} images prepared in {remote_dir}/{subset_folder}/")
    return len(images)


def create_config(
    name: str,
    trigger_word: str,
    dim: int = 16,
    epochs: int = 15,
    batch_size: int = 2,
) -> str:
    """Generate training TOML config and sample prompts, upload to K12.

    Args:
        name: Training name.
        trigger_word: Trigger word (used in sample prompts).
        dim: LoRA rank dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Config filename (e.g. "miki_v2_config.toml").
    """
    output_name = f"{name}_prodigy_dim{dim}"
    config_filename = f"{name}_config.toml"
    sample_filename = f"{name}_sample_prompts.txt"

    # --- TOML config (Prodigy template for NoobAI-XL-Vpred) ---
    toml_content = f"""[sdxl_arguments]
cache_text_encoder_outputs = true

[model_arguments]
pretrained_model_name_or_path = "{K12_CHECKPOINT}"
v_parameterization = true
zero_terminal_snr = true

[dataset_arguments]
train_data_dir = "{K12_DATASET_BASE}/{name}"
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 2048
bucket_reso_steps = 64
caption_extension = ".txt"

[training_arguments]
output_dir = "{K12_OUTPUT_DIR}"
output_name = "{output_name}"
save_model_as = "safetensors"
save_precision = "bf16"
save_every_n_epochs = 5
max_train_epochs = {epochs}
train_batch_size = {batch_size}
mixed_precision = "bf16"
gradient_checkpointing = true
gradient_accumulation_steps = 1
max_data_loader_n_workers = 4
persistent_data_loader_workers = true
seed = 42
prior_loss_weight = 1.0
max_token_length = 225
xformers = false
sdpa = true
logging_dir = "{K12_OUTPUT_DIR}/logs"
log_prefix = "{name}"
sample_sampler = "euler"
sample_every_n_epochs = 5
sample_prompts = "/config/{sample_filename}"

[optimizer_arguments]
optimizer_type = "Prodigy"
learning_rate = 1.0
optimizer_args = [
    "decouple=True",
    "weight_decay=0.01",
    "betas=0.9,0.99",
    "use_bias_correction=True",
    "safeguard_warmup=True",
    "d_coef=2.0",
]
lr_scheduler = "cosine"
lr_warmup_steps = 0

[network_arguments]
network_module = "networks.lora"
network_dim = {dim}
network_alpha = {dim // 2 if dim > 1 else 1}
network_train_unet_only = true
"""

    # --- Sample prompts ---
    sample_content = f"""{trigger_word}, masterpiece, best quality, 1girl, solo, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy --w 1024 --h 1024 --l 5.5 --s 28
{trigger_word}, masterpiece, best quality, 1girl, solo, full body, standing, smile, outdoors --n low quality, worst quality, bad anatomy --w 768 --h 1152 --l 5.5 --s 28
"""

    with tempfile.TemporaryDirectory(prefix="lora_cfg_") as tmpdir:
        toml_path = os.path.join(tmpdir, config_filename)
        sample_path = os.path.join(tmpdir, sample_filename)

        with open(toml_path, "w") as f:
            f.write(toml_content)
        with open(sample_path, "w") as f:
            f.write(sample_content)

        # Ensure remote config dir exists
        print(f"[create_config] Uploading config to {K12_HOST}:{K12_CONFIG_DIR}/")
        _run(["ssh", K12_HOST, "mkdir", "-p", K12_CONFIG_DIR])
        _run(["scp", toml_path, sample_path, f"{K12_HOST}:{K12_CONFIG_DIR}/"])

    print(f"[create_config] Done. Config: {config_filename}, Samples: {sample_filename}")
    return config_filename


def submit_job(name: str, config_filename: str) -> str:
    """Generate and apply a k3s Job for LoRA training.

    Args:
        name: Training name. Job will be named ``lora-{name}``.
        config_filename: Config file name returned by ``create_config()``.

    Returns:
        Job name (e.g. "lora-miki_v2").
    """
    job_name = f"lora-{name}"

    # Sanitize for k8s (underscores not allowed in metadata.name)
    k8s_job_name = job_name.replace("_", "-").lower()

    job_yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {k8s_job_name}
  namespace: {NAMESPACE}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: lora-training
        training-name: {name}
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: {DOCKER_IMAGE}
          imagePullPolicy: Never
          command:
            - python
            - -m
            - sdxl_train_network
            - --config_file
            - /config/{config_filename}
          resources:
            requests:
              memory: "12Gi"
              nvidia.com/gpu: "1"
            limits:
              memory: "16Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: checkpoints
              mountPath: /data/comfyui/models/checkpoints
              readOnly: true
            - name: datasets
              mountPath: {K12_DATASET_BASE}
              readOnly: true
            - name: config
              mountPath: /config
              readOnly: true
            - name: output
              mountPath: {K12_OUTPUT_DIR}
      volumes:
        - name: checkpoints
          hostPath:
            path: /data/comfyui/models/checkpoints
            type: Directory
        - name: datasets
          hostPath:
            path: {K12_DATASET_BASE}
            type: Directory
        - name: config
          hostPath:
            path: {K12_CONFIG_DIR}
            type: Directory
        - name: output
          hostPath:
            path: {K12_OUTPUT_DIR}
            type: DirectoryOrCreate
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="lora_job_", delete=False) as f:
        f.write(job_yaml)
        yaml_path = f.name

    try:
        # Delete existing job if any (ignore errors)
        print(f"[submit_job] Cleaning up any existing job '{k8s_job_name}'...")
        _run(
            ["kubectl", "delete", "job", k8s_job_name, "-n", NAMESPACE, "--ignore-not-found"],
            check=False,
        )

        # Apply the job
        print(f"[submit_job] Submitting job '{k8s_job_name}' to k3s...")
        _run(["kubectl", "apply", "-f", yaml_path])
    finally:
        os.unlink(yaml_path)

    print(f"[submit_job] Done. Job '{k8s_job_name}' submitted.")
    return k8s_job_name


def check_status(name: str) -> str:
    """Check the status of a LoRA training job.

    Args:
        name: Training name.

    Returns:
        Status string with job status, pod status, and recent logs.
    """
    k8s_job_name = f"lora-{name}".replace("_", "-").lower()
    lines: list[str] = []

    # Job status
    lines.append(f"=== Job: {k8s_job_name} (namespace: {NAMESPACE}) ===")
    result = _run(
        ["kubectl", "get", "job", k8s_job_name, "-n", NAMESPACE, "-o", "wide"],
        check=False,
    )
    lines.append(result.stdout.strip() if result.stdout else "(no output)")
    if result.returncode != 0 and result.stderr:
        lines.append(f"Error: {result.stderr.strip()}")

    # Pod status
    lines.append("")
    lines.append("=== Pod Status ===")
    result = _run(
        ["kubectl", "get", "pods", "-n", NAMESPACE, "-l", f"job-name={k8s_job_name}", "-o", "wide"],
        check=False,
    )
    lines.append(result.stdout.strip() if result.stdout else "(no pods found)")

    # Recent logs
    lines.append("")
    lines.append("=== Recent Logs (last 30 lines) ===")
    result = _run(
        ["kubectl", "logs", f"job/{k8s_job_name}", "-n", NAMESPACE, "--tail=30"],
        check=False,
    )
    if result.stdout:
        lines.append(result.stdout.strip())
    elif result.stderr:
        lines.append(f"(no logs available: {result.stderr.strip()})")
    else:
        lines.append("(no logs available)")

    status_text = "\n".join(lines)
    print(status_text)
    return status_text


def deploy_lora(name: str, dim: int = 16) -> str:
    """Copy trained LoRA to ComfyUI models directory on K12.

    Args:
        name: Training name.
        dim: LoRA rank (used in output filename).

    Returns:
        LoRA filename as referenced in ComfyUI.
    """
    lora_filename = f"{name}_prodigy_dim{dim}.safetensors"
    src_path = f"{K12_OUTPUT_DIR}/{lora_filename}"
    dst_path = f"{K12_LORA_DIR}/{lora_filename}"

    print(f"[deploy_lora] Copying {src_path} -> {dst_path} on K12...")
    _run(["ssh", K12_HOST, "cp", src_path, dst_path])

    print(f"[deploy_lora] Done. LoRA available in ComfyUI as: {lora_filename}")
    return lora_filename


def train_lora(
    name: str,
    image_dir: str,
    trigger_word: str,
    caption_tags: str | None = None,
    repeat: int = 10,
    dim: int = 16,
    epochs: int = 15,
    batch_size: int = 2,
) -> str:
    """Convenience function: prepare_dataset -> create_config -> submit_job.

    Args:
        name: Training name.
        image_dir: Local directory containing training images.
        trigger_word: Trigger word for the LoRA.
        caption_tags: Optional comma-separated tags.
        repeat: Number of repeats (encoded in dataset subfolder name, default 10).
        dim: LoRA rank dimension.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        k3s Job name.
    """
    print(f"[train_lora] Starting LoRA training pipeline for '{name}'")
    print("=" * 60)

    # Step 1: Prepare dataset
    print("\n--- Step 1/3: Prepare Dataset ---")
    count = prepare_dataset(name, image_dir, trigger_word, caption_tags, repeat=repeat)
    print(f"  Images prepared: {count}")

    # Step 2: Create config
    print("\n--- Step 2/3: Create Config ---")
    config_filename = create_config(name, trigger_word, dim=dim, epochs=epochs, batch_size=batch_size)
    print(f"  Config file: {config_filename}")

    # Step 3: Submit job
    print("\n--- Step 3/3: Submit Job ---")
    job_name = submit_job(name, config_filename)
    print(f"  Job name: {job_name}")

    print("\n" + "=" * 60)
    print(f"[train_lora] Pipeline complete. Monitor with: check_status('{name}')")
    print(f"[train_lora] After training: deploy_lora('{name}', dim={dim})")
    return job_name
