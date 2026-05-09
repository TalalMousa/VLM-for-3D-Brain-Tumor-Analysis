# TriNeuro AI

TriNeuro AI is a local brain-MRI decision-support web application. It accepts uploaded MRI modalities, runs a tumor screening model, classifies tumor type when needed, segments the lesion, generates Grad-CAM++ and attention visualizations, extracts segmentation-based clinical measurements, and produces a structured MRI report.

> Important: TriNeuro AI is a research/educational decision-support system. It is not a clinical diagnostic tool and must not replace radiologist review, clinical correlation, or histopathology.

## What The System Does

The final website pipeline is:

1. Upload MRI files through the browser.
2. Preprocess non-BraTS MRI files with IXI-style shape normalization and skull stripping.
3. Run a binary 3D MRI classifier: `no_tumor` vs `tumor`.
4. If a tumor is detected, run a second 3D classifier: `glioma` vs `meningioma`.
5. If a tumor is detected, generate or load a segmentation mask.
6. Extract clinical measurements from the mask.
7. Render MRI slices, segmentation overlays, Grad-CAM++, guided attention, and a same-slice AI summary panel.
8. Generate a structured final report.
9. Save recent patient reports locally in the browser.

## Final Website Files

```text
TriNeuro_XAI_Web/
  index.html              Frontend layout
  styles.css              Dashboard styling
  app.js                  Frontend logic and API calls
  api.py                  FastAPI backend and model inference pipeline
  assets/
    trineuro-logo.svg     TriNeuro AI logo

Report Generation Files/
  module2_feature_formatter.py   Segmentation-to-clinical-measurement formatter
  module4_medgemma_report.py     Concise report-generation workflow for explanation/testing

Final_Resnet_Tumor_NoTumor_93_ACC/
  NoTumor vs Tumor 93_.pt        Binary tumor/no-tumor checkpoint
  *.png, *.json                  Kept results, plots, Grad-CAM examples, split manifest

mask_guided_resnet18_gated_attention_raw_cache112/
  best_mask_guided_resnet18_gated_attention_t1c_t2f_raw112.pt
  *.png, *.csv, *.json           Kept metrics, plots, Grad-CAM examples, training history

Segmentation/
  seg_model.pth                  nnU-Net segmentation checkpoint
  *.png, *.json, *.txt           Kept progress, debug, mirror state, training logs

requirements.txt                 Final runtime dependencies
README.md                        This file
```

The codebase was cleaned for the final website. Training notebooks, cache builders, old CLI wrappers, Python bytecode caches, temporary logs, and unused report-generation utilities were removed. Model checkpoints, metrics, plots, images, CSV/JSON outputs, and training logs were kept.

## Models

### 1. Tumor Screening Model

Location:

```text
Final_Resnet_Tumor_NoTumor_93_ACC/NoTumor vs Tumor 93_.pt
```

Purpose:

```text
no_tumor vs tumor
```

Backend implementation:

```python
build_binary_model()
predict_binary()
preprocess_binary()
```

Input shape:

```text
1 x 128 x 128 x 128
```

The model is a 3D MONAI ResNet-style classifier using a single MRI volume.

### 2. Tumor Type Classifier

Location:

```text
mask_guided_resnet18_gated_attention_raw_cache112/
  best_mask_guided_resnet18_gated_attention_t1c_t2f_raw112.pt
```

Purpose:

```text
glioma vs meningioma
```

Backend implementation:

```python
MaskGuidedResNet18GatedAttention
predict_type()
preprocess_type()
compute_gradcam()
compute_learned_attention()
```

Input shape:

```text
2 x 112 x 112 x 112
```

Input modalities:

```text
T1c + T2F/FLAIR
```

During training, segmentation masks were used as attention guidance. During final website inference, the classifier receives MRI only.

### 3. Segmentation Model

Location:

```text
Segmentation/seg_model.pth
```

Purpose:

```text
Tumor segmentation for measurement and visualization
```

Backend implementation:

```python
segmentation_model()
preprocess_segmentation()
predict_segmentation()
```

Input shape:

```text
4 x 128 x 160 x 112
```

Expected modalities:

```text
T1n, T1c, T2w, T2-FLAIR
```

If some modalities are missing, the backend reuses the primary MRI so the pipeline can still run from limited input.

## Non-BraTS Preprocessing

Non-BraTS preprocessing was kept in the final website pipeline.

The frontend uploads files normally through `FormData`. The backend then applies:

```python
preprocess_non_brats_uploads(files, folder)
```

inside `/predict` before model inference.

The non-BraTS path includes:

```python
ixi_step1_register_or_shape_normalize()
ixi_step2_skull_strip()
estimate_ixi_brain_mask()
preprocess_non_brats_mri()
```

Behavior:

- BraTS-looking filenames are skipped.
- Segmentation masks are skipped.
- Non-BraTS MRI files are shape-normalized and skull-stripped.
- If `TRINEURO_IXI_TEMPLATE` points to a valid SRI24 template and `ants` is installed, rigid registration is attempted.
- If registration is unavailable, the backend falls back to canonical loading, center crop/pad, and local skull stripping.

The API returns `preprocessing_info` so the frontend can show whether IXI preprocessing was applied.

## Report Generation

The final website uses OpenAI `gpt-4.1` for report generation. The model is fixed in code and is not selected through the UI.

Add your API key to the root `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

Restart the FastAPI server after editing `.env`.

If the key is missing or the OpenAI call fails, the backend returns the structured local fallback report with a note explaining the problem.

The report includes:

- patient ID
- lesion volume
- technique
- comparison
- findings
- impression
- recommendations

Report section titles are rendered in bold in the frontend preview:

```text
Patient:
Volume:
TECHNIQUE:
COMPARISON:
FINDINGS:
IMPRESSION:
RECOMMENDATIONS:
```

The downloaded report remains plain text.

### Module 2: Clinical Feature Formatting

File:

```text
Report Generation Files/module2_feature_formatter.py
```

This converts segmentation masks into structured findings:

- tumor presence
- tumor voxel count
- lesion volume in cm3
- tumor core volume in cm3
- centroid
- hemisphere/laterality estimate
- bounding box size
- maximum diameter
- axial span
- surface area
- sphericity
- subregion volumes

### Module 4: Report Workflow For Explanation

File:

```text
Report Generation Files/module4_medgemma_report.py
```

This restored concise module is used by the final website report path and is also useful for project explanation/testing. It demonstrates the report-generation workflow:

1. draft report generation
2. critique
3. refinement

It also contains:

- `generate_mri_report_with_graph(...)`
- `build_report_graph(...)`
- `generate_medgemma_report(...)` compatibility helper
- `OpenAILLM` wrapper fixed to `gpt-4.1`

The website always uses `gpt-4.1` when `OPENAI_API_KEY` is present.

## Visualizations

The website returns and displays:

- interactive MRI slice preview
- segmentation overlay preview
- same-slice AI panel
- Grad-CAM++ heatmap
- guided attention map

The full AI panel is organized as:

```text
Rows:
  Sagittal
  Coronal
  Axial

Columns:
  T1c
  T2F / FLAIR
  Segmentation
  Grad-CAM++ + Segmentation
  Guided Attention
```

The Grad-CAM++ + Segmentation column overlays the Grad-CAM heatmap and the segmentation outline on the same slice.

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "TriNeuro AI API",
  "device": "cpu"
}
```

The device may be `cuda` when a GPU is available.

### Prediction

```http
POST /predict
```

Multipart form fields:

| Field | Required | Description |
|---|---:|---|
| `primary` | yes | Main MRI upload |
| `patient_id` | no | Patient/case ID |
| `force_tumor_branch` | no | Force tumor-type, segmentation, and visualization branch |
| `fast_mode` | no | Use smaller visualization previews |
| `include_full_panel` | no | Return full same-slice AI panel |
| `include_xai_slices` | no | Return Grad-CAM/attention slice arrays |
| `t1n` | no | T1 native modality |
| `t1c` | no | T1 contrast modality |
| `t2w` | no | T2 weighted modality |
| `t2f` | no | T2-FLAIR modality |
| `mask` | no | Optional uploaded segmentation mask |

Main response fields:

```json
{
  "tumor_result": {},
  "tumor_type_result": {},
  "preprocessing_info": {},
  "segmentation_source": "Segmentation/seg_model.pth",
  "clinical_findings": {},
  "report": "...",
  "report_source": "openai_gpt_4_1",
  "performance": {},
  "mri_slices": {},
  "segmentation_slices": {},
  "gradcam_slices": null,
  "attention_slices": null,
  "xai_panel_png": "...",
  "xai_full_panel_png": "...",
  "mri_column_png": null
}
```

## Frontend Behavior

File:

```text
TriNeuro_XAI_Web/app.js
```

The frontend:

- collects uploaded MRI files
- sends them to `http://localhost:8000/predict`
- requests `fast_mode=true`
- requests the full same-slice AI panel
- renders report text with bold section headers
- draws MRI and segmentation previews on matching canvases
- stores recent patient reports in browser `localStorage`
- allows report download as `.txt`

## Installation

Use Python 3.10+ if possible. The current machine has used Python 3.13 successfully for syntax/import checks, but medical-imaging packages are often smoother on Python 3.10 or 3.11.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Final runtime dependencies:

```text
numpy
nibabel
scipy
torch
monai
matplotlib
pillow
python-dotenv
openai
fastapi
uvicorn
python-multipart
nnunetv2
```

Optional:

- `ants` / ANTsPy for template-based IXI registration.

Required for final report generation:

- Add `OPENAI_API_KEY` to the root `.env` file.

## Running The Website

Set your OpenAI key in the root `.env` file before starting the backend:

```text
OPENAI_API_KEY=your_api_key_here
```

Open one terminal for the backend:

```powershell
cd "C:\Users\Basel\Downloads\Finished_Talal_GP _)\TriNeuro_XAI_Web"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open a second terminal for the frontend:

```powershell
cd "C:\Users\Basel\Downloads\Finished_Talal_GP _)\TriNeuro_XAI_Web"
python -m http.server 8080 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8080/
```

The frontend sends requests to:

```text
http://localhost:8000/predict
```

## Expected Upload Workflow

Minimum:

```text
Primary MRI
```

Recommended for best tumor branch results:

```text
Primary MRI
T1n
T1c
T2w
T2F / FLAIR
```

Optional:

```text
Segmentation mask
```

If a mask is uploaded, the backend uses the uploaded mask instead of generating one with `seg_model.pth`.

## Performance Notes

The first prediction can be slow because model checkpoints are loaded into memory:

- binary classifier
- tumor-type classifier
- segmentation model

After the first run, model loaders use `lru_cache`, so later requests avoid reloading the same models.

The frontend uses fast mode by default:

- fewer preview slices
- smaller PNG previews
- no unused MRI column image
- no Grad-CAM/attention slice arrays unless explicitly requested
- report generation still uses OpenAI `gpt-4.1`

## Important Accuracy Notes

- Segmentation accuracy depends on uploaded modalities being from the same study and spatially aligned.
- If modalities have different orientations, fields of view, or registration, segmentation overlays may be inaccurate.
- The UI preview alignment was adjusted so MRI and segmentation are drawn with matching canvas logic and the same selected slice index.
- Tumor type classification uses MRI only at inference time; masks guide explanation/reporting but are not tumor-type model inputs.
- The report is generated from model outputs and segmentation measurements, not from a radiologist.

## What Was Removed During Cleanup

The final website cleanup removed files that are not needed at runtime:

- training notebooks
- preprocessing notebooks
- report-generation notebook
- old cache-building utilities
- old CLI pipeline wrappers
- Python `__pycache__`
- temporary local server logs
- old `.env` used by removed report-generation path

The following were intentionally kept:

- model checkpoints
- image outputs
- plots
- metrics
- CSV/JSON results
- training logs
- final website source code
- clinical measurement formatter
- concise Module 4 report workflow for explanation

## Quick Sanity Checks

Compile Python:

```powershell
python -m py_compile TriNeuro_XAI_Web\api.py "Report Generation Files\module2_feature_formatter.py" "Report Generation Files\module4_medgemma_report.py"
```

Check JavaScript:

```powershell
node --check TriNeuro_XAI_Web\app.js
```

Check FastAPI import:

```powershell
cd "C:\Users\Basel\Downloads\Finished_Talal_GP _)\TriNeuro_XAI_Web"
python -c "from api import app; print(app.title)"
```

Expected output:

```text
TriNeuro AI API
```

## Project Summary

TriNeuro AI combines:

- 3D tumor/no-tumor screening
- 3D glioma/meningioma classification
- nnU-Net-style segmentation
- Grad-CAM++ explainability
- guided attention visualization
- segmentation-derived clinical measurements
- structured MRI report generation
- local browser-based dashboard

It is designed as a complete, local demonstration of a brain-MRI AI pipeline from upload to visualization and report.
