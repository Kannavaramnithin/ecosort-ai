# EcoSort AI - Intelligent Waste Classification Model ♻️

EcoSort AI is a deep learning computer vision model built using PyTorch and ResNet-50 architecture to automate municipal, household, and industrial waste classification into recyclable, biodegradable, and hazardous streams.

---

## 📌 Supported Waste Classes (10 Categories)

The model classifies waste images into 10 distinct categories:
1. `metal`
2. `glass`
3. `biological`
4. `paper`
5. `battery`
6. `trash`
7. `cardboard`
8. `shoes`
9. `clothes`
10. `plastic`

---

## 🏗️ Model Architecture

- **Base Architecture**: ResNet-50 (`torchvision.models.resnet50`)
- **Classification Head**: Customized Fully Connected (`nn.Linear(2048, 10)`)
- **Input Resolution**: 224 × 224 pixels (RGB)
- **Normalization**: ImageNet standards
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

---

## 📁 Repository Structure

```text
ecosort-ai/
├── README.md
└── garbage_classifier_model/
    ├── test_garbage_model.py     # End-to-end inference script with class confidence breakdown
    ├── verify_model.py           # Verification script to validate checkpoint weights
    ├── rebuild_model.py          # Script to package tensor shards into a .pth archive
    ├── data.pkl                  # PyTorch model state dictionary metadata
    ├── version                   # Serialization version identifier
    ├── byteorder                 # Byteorder metadata
    ├── .format_version           # Format version
    ├── .storage_alignment        # Storage alignment settings
    ├── .data/                    # Serialization ID
    └── data/                     # Raw serialized PyTorch tensor weights (layers & bias buffers)
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Dependencies

Ensure Python 3.10+ is installed along with required machine learning libraries:

```bash
pip install torch torchvision pillow
```

### 2. Rebuild the `.pth` Checkpoint (Optional)

If you need a single consolidated `best_model.pth` file from the repository tensor shards:

```bash
python garbage_classifier_model/rebuild_model.py
```

### 3. Verify Model Integrity

Check if the model checkpoint loads properly into PyTorch:

```bash
python garbage_classifier_model/verify_model.py
```

### 4. Run Inference on a Test Image

To test the model on a sample image:

```bash
python garbage_classifier_model/test_garbage_model.py
```

---

## 📊 Sample Inference Output

```text
==============================
       ECOSORT AI
==============================
metal           : 0.24%
glass           : 0.00%
biological      : 0.09%
paper           : 0.00%
battery         : 0.02%
trash           : 91.36%
cardboard       : 5.57%
shoes           : 1.56%
clothes         : 1.12%
plastic         : 0.04%
==============================
Prediction : trash
Confidence : 91.36%
==============================
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## 📄 License

This project is licensed under the MIT License.
