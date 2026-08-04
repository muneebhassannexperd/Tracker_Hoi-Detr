# Tracker_Hoi-Detr (HybridSORT)

HOI detection + multi-object tracking for hand–object interactions.

This repo combines:

- **[HOI-DETR](https://github.com/AhmadDarKhalil/HOI-DETR)** — detects hands, 1st objects, 2nd objects, and pairwise interactions
- **[HybridSORT](https://github.com/ymzis69/HybridSORT)** — tracks instances across video frames (Hybrid-SORT / Hybrid-SORT-ReID)

Per-class trackers keep IDs separate for `hand`, `firstobject`, and `secondobject`. Deep Hybrid SORT uses HOI-DETR decoder embeddings as appearance features (no separate ReID checkpoint).

---

## What’s included

| Piece | Path |
|-------|------|
| Video demo + tracking | `demo/demo_video.py` |
| Image demo (detection only) | `demo/demo.py` |
| Tracker adapters | `demo/hoi_trackers/` |
| Vendored HybridSORT | `external/HybridSORT/` |
| HOI-DETR / Co-DETR / MMDet | `projects/`, `mmdet/`, `configs/` |

Local artifacts are gitignored: `checkpoints/`, `Test-Data/`, `Output/`, caches, etc.

---

## Installation

Tested with Python **3.7**, PyTorch **1.11+cu113**, mmcv-full **1.5.0** (x86 NVIDIA GPU). For Hopper / aarch64 see [INSTALL_HOPPER.md](INSTALL_HOPPER.md). Detailed x86 notes: [INSTALL.md](INSTALL.md).

```bash
conda create -n codetr python=3.7 -y
conda activate codetr

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# from this repo root
pip install -e .
pip install timm==0.6.13 fairscale==0.4.6 scipy==1.7.3 yapf==0.40.1 \
  opencv-python numpy==1.21.6 pycocotools tqdm

# HybridSORT association deps
pip install lap filterpy
```

### C-BIoU env (separate — required)

Roboflow `trackers` needs **Python ≥ 3.10**, so it cannot live in `codetr`. Use a second env:

```bash
conda create -n cbiou python=3.10 -y
conda activate cbiou
pip install git+https://github.com/roboflow/trackers.git opencv-python-headless tqdm
```

You **cannot** run HOI-DETR inside `cbiou` without reinstalling the full detector stack. Compare trackers with a two-step workflow instead (below).


Verify:

```bash
python -c "import torch, mmcv; print(torch.__version__, mmcv.__version__)"
# Expected: 1.11.0+cu113  1.5.0
```

---

## Checkpoint

Weights are **not** in the repo (~5.5GB). Download into `checkpoints/`:

```bash
mkdir -p checkpoints
wget -O checkpoints/epoch_5.pth \
  https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth
```

---

## Tracked video demo

Edit settings at the top of `demo/demo_video.py`:

```python
MODEL_CONFIG = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py'
CHECKPOINT   = 'checkpoints/epoch_5.pth'
DEVICE       = 'cuda:0'
INPUT_DIR    = 'path/to/videos'   # searched recursively
OUTPUT_DIR   = 'Output'

TRACKER = 'hybrid_sort_reid'      # or 'hybrid_sort' / 'cbiou' / None
TRACK_ALLOW_BANK_RECLAIM = False  # HybridSORT: do not reuse passed-object IDs
TRACK_SPLIT_ON_CHANGE = True      # HybridSORT: split on size jump (shelf → product)

# When TRACKER = 'cbiou' (Roboflow; Python >= 3.10):
# CBIOU_BUFFER_RATIO_FIRST = 0.1
# CBIOU_BUFFER_RATIO_SECOND = 0.3
```

Run:

```bash
conda activate codetr
cd /path/to/Hoi-Detr_Hybrid_Sort
export PYTHONPATH=".:$PYTHONPATH"
python demo/demo_video.py
```

Outputs per video:

- annotated `.mp4`
- optional `.json` with per-frame detections, `track_id`, and interactions (`EXPORT_JSON = True`)

### Tracker options

| `TRACKER` value | Behavior |
|-----------------|----------|
| `hybrid_sort_reid` / `deep_hybrid_sort` | Hybrid-SORT-ReID using HOI embeddings |
| `hybrid_sort` | Hybrid-SORT (TCM / weak cues, no ReID) |
| `cbiou` | Use offline path below (`demo/track_cbiou_offline.py` in env `cbiou`) |
| `None` / `'none'` | Detection + interaction only (no IDs) |

### Side-by-side: HybridSORT vs C-BIoU (two envs)

One Python process = one conda env. Keep detection in `codetr`, re-track with C-BIoU in `cbiou`:

```bash
# 1) Detect (+ optional HybridSORT) in codetr → writes Output/<name>.json
conda activate codetr
export PYTHONPATH=".:$PYTHONPATH"
# TRACKER = 'hybrid_sort_reid'  or  None   in demo_video.py
python demo/demo_video.py

# 2) Replace track_ids with C-BIoU (same boxes / interactions)
conda activate cbiou
# edit INPUT_JSON in demo/track_cbiou_offline.py
python demo/track_cbiou_offline.py
# → writes Output/<name>_cbiou.json

# 3) Re-render the C-BIoU JSON (draw_ui lives with HOI helpers)
conda activate codetr
# set PREDICTIONS_JSON to the *_cbiou.json in demo/vis_offline.py
python demo/vis_offline.py
```

Tune `CBIOU_BUFFER_RATIO_FIRST` / `CBIOU_BUFFER_RATIO_SECOND` in `demo/track_cbiou_offline.py` (and the matching knobs in `demo_video.py` if you later run live C-BIoU on Python ≥ 3.10).


### ID stability (defaults)

- Stable display IDs are **not** copied from Hybrid’s internal Kalman counter
- Bank reclaim of old IDs is **off** by default (`TRACK_ALLOW_BANK_RECLAIM = False`)
- Large area / appearance jumps on a continuing tracklet mint a **new** ID
- Brief dropouts can keep an ID via last-frame continuity only

---

## Image demo

```bash
export PYTHONPATH=".:$PYTHONPATH"
python demo/demo.py
```

Edit `INPUT_DIR` / `OUTPUT_DIR` in `demo/demo.py`. Re-render from saved JSON with `demo/vis_offline.py`.

---

## Exported JSON (video)

Each detection may include `track_id` when tracking is enabled:

```jsonc
{
  "box": [x1, y1, x2, y2],
  "score": 0.97,
  "class_id": 0,
  "class_name": "hand",
  "track_id": 3
}
```

Interactions (`hf`, `fs`) reference detection indices. Full schema: `demo/predictions_io.py`.

---

## Project layout

```
Hoi-Detr_Hybrid_Sort/
├── demo/
│   ├── demo.py              # image HOI demo
│   ├── demo_video.py        # video HOI + tracking
│   ├── hoi_trackers/        # HybridSORT adapter + stable IDs
│   ├── helpers.py
│   └── predictions_io.py
├── external/HybridSORT/     # vendored tracker code
├── projects/                # HOI-DETR / Co-DETR models & configs
├── mmdet/                   # MMDetection (vendored)
├── checkpoints/             # put epoch_5.pth here (gitignored)
├── Test-Data/               # local videos (gitignored)
└── Output/                  # run outputs (gitignored)
```

---

## Acknowledgements

- Detection / interaction: [HOI-DETR](https://github.com/AhmadDarKhalil/HOI-DETR) ([paper](https://arxiv.org/abs/2606.17384))
- Base detector: [Co-DETR](https://github.com/Sense-X/Co-DETR) / [MMDetection](https://github.com/open-mmlab/mmdetection)
- Tracking: [HybridSORT](https://github.com/ymzis69/HybridSORT)

## License

See [LICENSE.txt](LICENSE.txt) and the licenses of upstream projects.
