# Research code: SCNN brain-MRI study and app model

Reproduces the analyses in the two manuscripts.

| Step | Command | Output |
| --- | --- | --- |
| 1. Data | `python prepare_data.py` | `data/prepared.npz`, `data/manifest.csv`, `data/dedup_report.json` |
| 2. Grid search (validation only) | `python grid_search.py --chunk 0 --of 2` then `--chunk 1 --of 2` | `runs/grid_results.csv` (108 rows) |
| 3. Test evaluation (Manuscript A) | `python evaluate.py` | `runs/test_results.json`, `runs/roc.png`, `runs/cv_results.csv` |
| 4. App model + TF.js export (Manuscript B) | `python train_app_model.py` | `runs/tfjs_mri/` → copy to `public/models/mri/` |
| 5. TF.js check | `node verify_tfjs.mjs runs/tfjs_mri <n> 64 4` | agreement with Keras, latency, Grad-CAM |
| 6. Grad-CAM box sizes | `python gradcam_stats.py` | median box area at each threshold |
| 7. Grad-CAM figure | `python gradcam_figure.py` | `runs/gradcam_examples.png` |

Data: Sartaj Bhuvaji, Brain Tumor Classification (MRI), https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet
(download the repository zip to `data/` and unzip). Seed 20261024. Python 3.13, TensorFlow 2.21, Keras 3.15.
`data/` and `runs/` are git-ignored. The results reported in the papers are copied to `results/`.
