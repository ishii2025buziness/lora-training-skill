---
name: lora-training
description: K12上でLoRA学習を実行するスキル。データセット準備・設定生成・ジョブ投入・デプロイを標準化。
triggers:
  - LoRA学習
  - LoRA training
  - train lora
  - 学習して
allowed-tools:
  - Bash(kubectl *)
  - Bash(scp *)
  - Bash(ssh k12 *)
  - Bash(uv run *)
  - Bash(ls *)
  - Bash(mkdir *)
---

# lora-training

K12 (NixOS homelab, RTX 5060 Ti 16GB, k3s) 上で LoRA 学習を実行するスキル。

## Quick Reference

```python
from lib.lora_training import train_lora, check_status, deploy_lora

# 一括実行: データセット準備 → 設定生成 → ジョブ投入
job_name = train_lora(
    name="miki_v2",
    image_dir="/home/kento/work/miki_images",
    trigger_word="hoshii_miki",
    caption_tags="1girl, blonde hair, green eyes",
    dim=16,
    epochs=15,
    batch_size=2,
)

# ジョブの状態確認
status = check_status("miki_v2")
print(status)

# 学習完了後、ComfyUI にデプロイ
lora_filename = deploy_lora("miki_v2", dim=16)
print(f"ComfyUI LoRA name: {lora_filename}")
```

## Individual Functions

### `prepare_dataset(name, image_dir, trigger_word, caption_tags=None, repeat=10)`

ローカルの画像を K12 にアップロードし、キャプションファイルを生成する。
sd-scripts が要求する `{repeat}_{trigger_word}/` サブフォルダ構造を自動作成する。

K12 上のディレクトリ構造:
```
/home/kento/lora-training/datasets/{name}/
    {repeat}_{trigger_word}/
        image1.png
        image1.txt
        ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | データセット名 (ディレクトリ名に使用) |
| `image_dir` | str | required | ローカルの画像ディレクトリパス |
| `trigger_word` | str | required | トリガーワード |
| `caption_tags` | str | None | 追加タグ (カンマ区切り)。None の場合 trigger_word のみ |
| `repeat` | int | 10 | リピート回数 (サブフォルダ名に使用) |

Returns: `int` — アップロードした画像数

### `create_config(name, trigger_word, dim=16, epochs=15, batch_size=2)`

Prodigy テンプレートから学習設定 TOML を生成し K12 にアップロード。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | 学習名 |
| `trigger_word` | str | required | トリガーワード (サンプルプロンプトに使用) |
| `dim` | int | 16 | LoRA rank (dim) |
| `epochs` | int | 15 | エポック数 |
| `batch_size` | int | 2 | バッチサイズ |

Returns: `str` — 設定ファイル名

### `submit_job(name, config_filename)`

k3s に学習ジョブを投入する。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | 学習名 (Job名: `lora-{name}`) |
| `config_filename` | str | required | `create_config` が返した設定ファイル名 |

Returns: `str` — Job 名

### `check_status(name)`

ジョブの状態・Pod ステータス・直近のログを取得する。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | 学習名 |

Returns: `str` — ステータス情報

### `deploy_lora(name, dim=16)`

学習済み LoRA を ComfyUI の models ディレクトリにコピーする。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | 学習名 |
| `dim` | int | 16 | LoRA rank (出力ファイル名に使用) |

Returns: `str` — ComfyUI で参照する LoRA ファイル名

### `train_lora(name, image_dir, trigger_word, caption_tags=None, repeat=10, dim=16, epochs=15, batch_size=2)`

`prepare_dataset` → `create_config` → `submit_job` を順に実行する便利関数。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | 学習名 |
| `image_dir` | str | required | ローカルの画像ディレクトリパス |
| `trigger_word` | str | required | トリガーワード |
| `caption_tags` | str | None | 追加タグ |
| `repeat` | int | 10 | リピート回数 |
| `dim` | int | 16 | LoRA rank |
| `epochs` | int | 15 | エポック数 |
| `batch_size` | int | 2 | バッチサイズ |

Returns: `str` — Job 名

## Architecture

- K12 へのファイル転送: `scp` (hostname: `k12`)
- K12 上のコマンド実行: `ssh k12 ...`
- k3s 操作: `kubectl` (G3 の ~/.kube/config 経由、直接実行)
- Docker イメージ: `localhost/lora-training:v1` (K12 上に構築済み)
- ベースモデル: NoobAI-XL-Vpred (`/data/comfyui/models/checkpoints/NoobAI-XL-Vpred-v1.0.safetensors`)
- Namespace: `lora-training` (作成済み)
