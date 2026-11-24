# EEC174 AY Mini-Project A2 Report  
**Image Mask Manipulations and Path Tracking**

---

## 1. Project Overview

This assignment has two phases:

1. **Phase 1 — Image Mask Manipulation** (`A2_Phase1.ipynb`): run DeepLabV3 semantic segmentation, then use the resulting masks for inpainting, per-instance stylization, and cross-image object replacement.
2. **Phase 2 — Path Tracking** (`tracker.py`, `yolo_path_tracker.py`): detect people with YOLOv3, associate detections across frames with a naive centroid tracker (SORT-like), and visualize IDs plus motion trails.

---

## 2. Workflow

```
Phase 1
  Images ──► DeepLabV3 ──► Class masks
                │
                ├─► Inpaint (mean / blur / Telea / patch)
                ├─► Style each connected-component instance
                └─► Replace: inpaint image1 + paste resized instance from image2

Phase 2
  Video frame ──► YOLO blob ──► detections ──► NMS
                                      │
                                      ▼
                               centroids (persons)
                                      │
                                      ▼
                         Tracker.update (nearest match ≤ 50 px)
                                      │
                                      ▼
                    draw boxes + ID labels + centroid trails ──► output.mp4
```

### Phase 1 pipeline (detail)

1. Load `image1` / `image2` (RGB).
2. Run DeepLabV3-ResNet50 once per image → integer class map.
3. **Inpainting:** build a binary mask for a chosen class, then fill that region with one of four methods.
4. **Styling:** for every non-background class, split instances with `cv2.connectedComponents`, assign a rotating style (pixelate, invert, cartoon, warm tint, edges, posterize).
5. **Replacement:** inpaint the class out of `image1`, pick a random same-class instance from `image2`, resize to the destination bbox, and paste with the instance mask.

### Phase 2 pipeline (detail)

1. Read each frame; form a 416×416 blob; forward YOLOv3.
2. Keep `person` detections above confidence 0.5; apply NMS (0.3).
3. Feed centroids into `Tracker.update`.
4. Draw bounding boxes, ID text, and short trails from stored paths.

---

## 3. Code Explanation (no source dump)

### 3.1 Phase 1 helpers

| Piece | Role |
|--------|------|
| `segmentation(image)` | Normalize → DeepLabV3 → argmax class map |
| `display_masks(...)` | Original image + one panel per detected class (no background) |
| `mean_color_fill` | Dilate mask, average (or majority) surrounding RGB, flat fill |
| `noise_fill` | Dilate + Gaussian feather; blend large blur into the hole |
| `linear_interpolation` | `cv2.inpaint` (Telea) from mask boundary |
| `patch_inpaint` | Copy nearby background patches into the hole |
| `apply_styles` | Unique OpenCV style per connected instance |
| `replace_object` | Inpaint destination + masked paste from source image |
| `show_intermediate` | Six-panel debug of the replacement stages |

### 3.2 Tracker (`tracker.py`)

Naive multi-object tracker driven by centroid proximity:

- **`register` / `deregister`:** allocate or free an ID and its centroid, lost counter, and path list.
- **`_distance`:** pairwise Euclidean distances between two centroid lists.
- **`update`:**
  - no detections → increment `lost`; drop IDs past `maxLost`;
  - no existing tracks → register all detections;
  - otherwise greedy nearest-neighbor match; reject pairs farther than **`MAX_DISTANCE = 50`**;
  - unmatched detections become new IDs; unmatched tracks age out.
- **`get_paths`:** returns per-ID lists of centroids (trimmed to a max length for clean trails).

### 3.3 YOLO path tracker (`yolo_path_tracker.py`)

Mirrors the A1 detect loop, but calls our `Tracker` instead of SORT and draws trails with `cv2.line` between consecutive path points.

---

## 4. Results

### Phase 1

- Segmentation masks correctly isolate major classes (e.g. Person) on both demo images.
- Side-by-side inpainting shows the expected tradeoffs: mean fill is flat; blur fill is soft; Telea inpaint is usually cleanest; patch fill keeps some texture but can tile.
- Styling produces visually distinct treatments per instance (≥ 4 styles).
- Replacement removes the class from `image1` and inserts a scaled instance from `image2`; intermediate panels confirm each step.

### Phase 2

- People in `input.webm` receive stable IDs while they stay within the 50 px match radius.
- Colored trails follow centroids; IDs and boxes are drawn on `output.mp4` (compare to `out_example/output.mp4`).
- Objects that leave the scene longer than `maxLost` frames are deregistered so IDs do not linger forever.

### Known limitations

- Centroid-only association can swap IDs under heavy occlusion (no Kalman / IoU).
- `MAX_DISTANCE = 50` may drop fast movers; raise carefully if needed.
- Patch inpainting is slower and more artifact-prone on large masks.
- Replacement assumes both images contain the target class and uses axis-aligned resize (no pose alignment).

---

## 5. Contributions

| Member | Contributions |
|--------|----------------|
| **Saarth Kishore** | Led Phase 1 (`A2_Phase1.ipynb`): segmentation visualization, inpainting methods, stylization, object replacement, and notebook Q&A |
| **Samarth Kale** | Led Phase 2 (`tracker.py`, `yolo_path_tracker.py`): YOLO detection loop, centroid tracker, path visualization, and output video |

We peer-reviewed each other’s work after the initial implementation (Saarth reviewed Phase 2; Samarth reviewed Phase 1), then used Cursor to help draft and organize this report.

---

## 6. How to Run

### Phase 1
Open `A2_Phase1.ipynb` in Google Colab or a local Jupyter environment with `torch`, `torchvision`, `opencv-python`, `matplotlib`, and `pillow`. Set `image1_path` / `image2_path` at the top, then run all cells.

### Phase 2
Place `yolov3.weights` in `yolo_files/` (same weights as A1), then:

```bash
python yolo_path_tracker.py
```

Output is written to `output.mp4`.

---

## 7. Files Submitted

- `A2_Phase1.ipynb`
- `tracker.py`
- `yolo_path_tracker.py`
- `REPORT.md` (this document)
