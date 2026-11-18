# Six-class NEU-DET dataset preparation and honest evaluation

This guide is for **Bharat AI-SoC PS5**, where a YOLOv8 Nano model has already been trained and exported as NCNN. **Do not retrain or create a new random test split at this stage.** The objective is to locate a genuinely held-out, annotated subset from the *original training run*, then evaluate the same deployed weights.

## 1. Source and annotation type

The official [NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) identifies **NEU-DET**, the detection version of the dataset, as the one with bounding-box annotations. It contains six steel defect categories:

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

The usual original layout includes **IMAGES/** with images and **ANNOTATIONS/** containing Pascal VOC **.xml** labels. Many third-party YOLO conversions instead have **train/images/**, **train/labels/**, **test/images/**, **test/labels/** and a **data.yaml**. Different re-packagings use different splits and class IDs. Never assume a universal official test split.

**NEU-CLS images without bounding-box labels are not suitable for AP50/mAP50 evaluation.**

## 2. Inventory without changing anything

On Raspberry Pi SSH (not Windows PowerShell), after pulling the new code:

\`\`\`bash
cd ~/Industrial_Defect_Detection
source .venv/bin/activate
python scripts/audit_dataset.py --roots datasets
\`\`\`

This prints:

- image and annotation-file counts
- actual matched image/label pairs
- explicit train/val/test folder counts, if any
- counts by XML class name or by YOLO numeric class ID
- examples of missing, ambiguous or invalid labels
- candidate original dataset folders, where present

This tool is **read-only**, doesn't run YOLO or Llama, and doesn't alter the existing database. An image under a directory named \`test\` is only a **test candidate**, not proof it was unseen during training.

If no labels are found, locate the original dataset on the training computer. On Windows, the project may be in a different workspace from the Pi deployment project. Inspect the local dataset and look for \`.xml\` or paired \`.txt\` labels and \`data.yaml\`. An official NEU-DET download with XML is one option, but only a preserved known held-out split from your own original training process is valid as a test for the already-trained model.

**Don't simply split all 1,800 NEU-DET images at random after training.** That can put previously seen training images into the evaluation set and inflate results.

## 3. Required folder layouts

### Pascal VOC XML example

\`\`\`text
datasets/processed/test/
  images/
    crazing_10.jpg
    inclusion_20.jpg
    ...
  labels/
    crazing_10.xml
    inclusion_20.xml
    ...
\`\`\`

The image and annotation names need the same stem. An XML's \`<object><name>\` labels are authoritative; the evaluator normalizes hyphens and underscores.

### YOLO TXT example

\`\`\`text
datasets/processed/test/
  images/
    crazing_10.jpg
    ...
  labels/
    crazing_10.txt
    ...
dataset.yaml  # or original data.yaml with 'names' in correct index order
\`\`\`

For **.txt** labels, the evaluator **requires the actual original dataset YAML**. Example:

\`\`\`yaml
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
\`\`\`

**This ordering is an illustration, not a claim about your trained dataset.** Use the names and indices in the actual data.yaml from your dataset. The NCNN model's class index order may differ.

## 4. Run a small smoke test, then full evaluation

Only after confirming the test folder is genuinely held out and the Pi has a reliable power supply:

**VOC XML example:**

\`\`\`bash
python scripts/evaluate_dataset.py \\
  --images datasets/processed/test/images \\
  --labels datasets/processed/test/labels \\
  --limit 6
\`\`\`

**YOLO TXT example:**

\`\`\`bash
python scripts/evaluate_dataset.py \\
  --images datasets/processed/test/images \\
  --labels datasets/processed/test/labels \\
  --data-yaml /path/to/your/original/data.yaml \\
  --limit 6
\`\`\`

Remove \`--limit 6\` for the final full-test evaluation. If your original training image folder is available, add \`--train-images /path/to/original/train/images\` to check for **exact byte-for-byte train/test duplicates**. This check cannot detect all leakage (e.g., re-encoded or augmented duplicates), so training provenance still matters.

**Report semantics:**

| Output | Meaning |
|---|---|
| Per-class precision/recall/F1 | At \`--operating-conf 0.30\` by default, matching the dashboard |
| AP50 | Area under the precision/recall curve at IoU 0.50, using \`--conf 0.001\` detections |
| mAP50 | Average AP50 over represented classes; six-class claim only if all six have ground truth |
| mAP50–95 | Not calculated by this script; use Ultralytics validation or another supported evaluator if required |
| Latency | YOLO inference time per image in the evaluation loop, not camera FPS |

Output paths (intentionally Git-ignored):

\`\`\`text
logs/evaluation/metrics.json
logs/evaluation/per_class_metrics.csv
\`\`\`

**Do not estimate metrics from one scratch image.** Save the complete test-image count, test-split source, class counts, date, model build and used parameters in the final report.

## 5. Power and implementation notes

- Your previous Pi 5 power source reported under-voltage/throttling. Avoid full 180-image or 360-image evaluations until supplied with reliable dedicated Pi 5 power.
- The current verified NCNN model should be evaluated as-is; changing the train split or fine-tuning requires a new explicit benchmark run.
- The six NEU-DET categories are *defects only*. Even an excellent mAP50 does not justify automated acceptance of defect-free steel.
- Camera hardware tests, repeatable latency/FPS, and calibrated severity remain separate future-development tasks.
