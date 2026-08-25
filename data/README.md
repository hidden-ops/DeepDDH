# Data directory

No ultrasound image, annotation, or patient-level metadata is distributed with this repository.

Place locally authorized data into the existing empty directories without changing the folder names:

```text
data/
├── Training/
│   ├── imgs/<case_id>.png
│   ├── segs/<case_id>.png
│   └── bony_masks/<case_id>.png
├── Testing/
│   ├── imgs/<case_id>.png
│   ├── segs/<case_id>.png
│   └── bony_masks/<case_id>.png
└── Ext-1/ ... Ext-8/                 # optional; same three subdirectories
```

Requirements:

- Each case must use the same filename stem in `imgs`, `segs`, and `bony_masks`.
- The current loaders accept PNG files only.
- Input images are expected to be RGB and all three files for a case must have the same spatial size.
- `segs` is an integer class-index mask with values 0–7 (background plus seven anatomical structures). The current STEP4 post-processing defaults to class 5 for the labrum; use `--labrum-class` if your annotation map uses another ID.
- `bony_masks` is an integer class-index mask with values 0–6 (background plus six iliac substructures).
- Grayscale masks are preferred. If an RGB mask is supplied, the current loader reads its first channel.
- Do not convert class-index masks into 0/255 binary masks or one-hot arrays.

The `.gitignore` file excludes all data-directory contents except this documentation, utility scripts, directory placeholders, and `.gitkeep` files. Verify `git status` before every public push.
