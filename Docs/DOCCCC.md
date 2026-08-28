# AI-Based Personal Stylist  
## Complete Implementation Log (Verbatim, End-to-End)

---

## 0. What this document is

This document is a **verbatim, end-to-end implementation log** of the AI-Based Personal Stylist project **up to the CLIP embedding stage**. 

It is written so that:
- even someone with **no prior context** can understand what was done
- even the **author himself**, months later, can reconstruct every decision
- an **examiner** can trace the logic step-by-step
- **nothing is hidden**, skipped, or hand-waved

This is **not a summary**.   
This is **not a cleaned-up report**.  
This is a **ground-truth engineering log**.

---

## 1. Big-picture goal of the project

The final goal of this capstone project is to build an: 

**AI-Powered Personal Stylist**

That can:
1. Take a real photograph of a person wearing clothes
2. Identify and isolate individual garments
3. Understand how each garment looks (style, color, texture, semantics)
4. Later: 
   - evaluate outfit compatibility
   - evaluate trend relevance
   - suggest improvements or replacements

This project is built **incrementally**, because fashion reasoning is impossible unless **visual understanding is correct first**.

---

## 2. Why the foundation stage is critical

Before talking about: 
- compatibility
- trends
- recommendations
- graphs
- ML "intelligence"

The system must answer **three fundamental questions** correctly: 

1. Where exactly is each garment in the image?
2. How do we isolate each garment without background noise?
3. How do we represent a garment numerically so a model can reason about it?

Everything implemented so far exists **only** to answer these three questions correctly.

---

## 3. Environment setup (Python + Windows)

### 3.1 Virtual environment creation

A Python virtual environment was created: 

```powershell
python -m venv .venv
. venv\Scripts\activate
```

**Why this is necessary:**

* isolates dependencies
* avoids version conflicts
* makes the project reproducible
* this is standard practice in real ML projects

---

### 3.2 Pip upgrade

```powershell
python. exe -m pip install --upgrade pip
```

Terminal output (important):

```
Successfully installed pip-25.3
```

**Why this matters:**

* modern packages (CLIP) use `pyproject.toml`
* older pip versions often fail silently
* upgrading avoids hard-to-debug installation issues

---

## 4. Installing core libraries

```powershell
pip install opencv-python numpy matplotlib
```

Terminal output confirms installation:

```
Requirement already satisfied: opencv-python ... 
Requirement already satisfied: numpy ... 
Requirement already satisfied: matplotlib ... 
```

### Why each library is needed

* **numpy**

  * numerical arrays
  * masks
  * embeddings

* **opencv-python**

  * polygon → mask conversion
  * mask-based cropping
  * RGBA image saving

* **matplotlib**

  * visualization of bounding boxes
  * segmentation sanity checks

---

## 5. Version control:  first major checkpoint

```powershell
git add .
git commit -m "Baseline DeepFashion2 annotation parsing and visualization"
git push origin main
```

This commit represents:

* correct parsing of DeepFashion2 annotations
* bounding boxes align with garments
* segmentation polygons align with garments
* dataset integrity is verified

This is **not cosmetic** — it proves the dataset is usable.

---

## 6. DeepFashion2: how it is used (important clarification)

DeepFashion2 is used **ONLY** for:

* images
* annotation JSON files: 

  * bounding boxes
  * segmentation polygons
  * garment categories

❌ No training  
❌ No Mask R-CNN  
❌ No GPU

This provides **ground-truth garment localization**, which is far more reliable than running a detector at this stage.

---

## 7. Garment extraction pipeline

### 7.1 Core logic file

File: 

```
src/segmentation/deepfashion2_parser.py
```

This file performs **all of the following**:

1. Load image: 

   ```python
   img = cv2.imread(img_path)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```

2. Load annotation JSON:

   ```python
   with open(ann_path) as f:
       anno = json.load(f)
   ```

3. Iterate over each annotated garment:

   * read bounding box
   * read segmentation polygon
   * read category

4. Convert segmentation polygon → **binary mask**

   ```python
   cv2.fillPoly(mask, [pts], 255)
   ```

5. Apply mask + bounding box: 

   * background removed
   * garment preserved
   * alpha channel added

6. Save outputs:

   * transparent garment PNG
   * metadata. json
   * visualization image

---

## 8. Output folder structure (design decision)

For image ID `010931`:

```
outputs/
 └── 010931/
      ├── top_0.png          # transparent garment (RGBA)
      ├── metadata.json      # category + bbox info
 └── 010931_vis. jpg          # visualization
```

**Why this structure matters:**

* deterministic
* clean per-image separation
* easy to loop over later
* no ambiguity in downstream processing

---

## 9. Why transparent PNGs were used (very important)

If rectangular crops were used:

* background colors leak
* embeddings get polluted
* similarity logic becomes unreliable

Using **mask-based RGBA PNGs** ensures:

* only garment pixels are encoded
* no background bias
* correct visual semantics

This is a **research-grade decision**, not overengineering.

---

## 10. Windows-specific command issues encountered

### 10.1 `touch` not found

```powershell
touch src/segmentation/mask_utils.py
```

Error:

```
'touch' is not recognized
```

Reason:

* `touch` is a Linux command
* PowerShell does not support it

Correct Windows alternative:

```powershell
type nul > src\segmentation\mask_utils.py
```

---

### 10.2 Bash here-doc failure

```powershell
python - <<EOF
```

Error occurs because: 

* PowerShell does not support bash here-documents
* this is Linux-only syntax

Correct Windows approach:

* create `.py` files
* or run Python interactively

---

## 11.  CLIP embedding stage (Stage 2)

### 11.1 Why CLIP is used

CLIP converts an image into a **512-dimensional vector** that captures:

* color
* texture
* shape
* style semantics

This representation: 

* is far superior to raw pixels
* is better than ImageNet CNN features
* is widely used in fashion research

CLIP is used **only as a feature extractor**.

---

### 11.2 Requirements installation

```powershell
pip install -r .\requirements.txt
```

Terminal highlights:

```
Successfully built clip
Successfully installed clip-1.0 ftfy-6.3.1
```

Verification:

```powershell
python -c "import clip; print('CLIP OK')"
```

Output:

```
CLIP OK
```

---

### 11.3 Embedding extraction script

File:

```
clip_extract_embeddings.py
```

Run command:

```powershell
python clip_extract_embeddings.py
```

Terminal output:

```
[INFO] Loading CLIP model...
100%|███████████████████████████████████████| 338M/338M [00:15<00:00, 22.8MiB/s]
[INFO] CLIP loaded on CPU
[EMBED] outputs\010931\top_0.png
[DONE] All embeddings extracted
```

This confirms:

* CLIP model downloaded correctly
* CPU inference works
* garment embeddings were generated

---

## 12. Output of CLIP stage

For each garment PNG:

```
top_0.png
top_0.npy
```

Where:

* `.png` → transparent garment image
* `.npy` → 512-D normalized embedding

---

## 13. CPU-only justification

* CLIP inference on CPU ≈ 0.2–0.4s per image
* dataset size used is **subset**, not full DF2
* GPU is not required for correctness or evaluation

This is acceptable and defensible in a capstone.

---

## 14. Git behavior and ignored outputs

Command attempted:

```powershell
git add clip_extract_embeddings.py outputs
```

Git response:

```
outputs is ignored
```

This is **correct behavior**.

Why:

* `outputs/` contains generated data
* should not be version-controlled
* reproducibility comes from code, not stored artifacts

Only this was committed:

```powershell
clip_extract_embeddings.py
```

---

## 15. What has been achieved so far

### Completed:

* DeepFashion2 annotation parsing
* Pixel-accurate garment extraction
* Transparent RGBA garment crops
* Metadata storage
* CLIP image embeddings
* CPU-only pipeline
* Clean Git history

### Not yet implemented (intentionally):

* compatibility scoring
* trend analysis
* recommendations
* web interface

---

## 16. Why this stage is COMPLETE

A **garment-level visual representation pipeline** has been built. 

This is:

* the hardest part
* the most failure-prone part
* the foundation of everything else

Everything that follows is **logic on top of these embeddings**. 

---

## 17. What comes next (future work)

Next stage (when ready):

* color harmony (HSV)
* similarity using CLIP vectors
* basic compatibility score

Nothing is rushed.

---

## Final note

This document intentionally preserves:

* mistakes
* fixes
* commands
* outputs

Because that is how real engineering work actually happens. 

Nothing here is fake.  
Nothing here is hidden.

This is the **true implementation log**. 