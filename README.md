<div align="center">

# Multi-time scale feature extraction and attention networks for automatic depression level prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Paper](https://img.shields.io/badge/Paper-Applied%20Soft%20Computing-red.svg)](https://doi.org/10.1016/j.asoc.2025.114052)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official source code of the paper: **"Multi-time scale feature extraction and attention networks for automatic depression level prediction"** published in *Applied Soft Computing* (Elsevier, 2026).

[Sarmad Al-Gawwam](https://scholar.google.com/), [Aleksandr Zaitcev](https://scholar.google.com/), [Mohammad R. Eissa](https://scholar.google.com/), [Noor Alshwilli](https://scholar.google.com/), [Mohammed Benaissa](https://scholar.google.com/)

*Department of Electronic and Electrical Engineering, University of Sheffield, Sheffield, S1 3JD, UK*

<br>
<img src="/model_page-0001.jpg" alt="MSFE-CTA Architecture" width="800">
<br>

</div>

---

## 📝 Abstract

Depression impairs functioning across personal and professional domains, and early detection is essential for timely intervention. Existing clinical assessments rely on specialists, limiting accessibility and scalability. This paper proposes an automated, video-based approach that estimates depression severity directly from full-length interviews. Facial markers evolve over micro- to macro- timescales; therefore, focusing solely on short clips risks missing long-range cues. 

This repository introduces the **Multi-Timescale Feature Extraction and Channel-Temporal Attention network (MSFE-CTA)** that learns dependencies across milliseconds, seconds, and minutes from complete recordings. The MSFE module employs stacks of Inception-TCN blocks with logarithmically scaled dilations to efficiently capture long-range structure, while the CTA module integrates dilated channel attention with multi-kernel depthwise temporal attention to highlight salient features.

Evaluated on the AVEC2013, AVEC2014, AVEC2017, and AVEC2019 datasets, MSFE-CTA achieves state-of-the-art performance with substantially lower computational cost (only 0.85 M parameters and 1.85 GFLOPs).

---

## ⚙️ Installation

To use this code, clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/msfe-cta-depression.git](https://github.com/your-username/msfe-cta-depression.git)
cd msfe-cta-depression
pip install -r requirements.txt
```

**Main Dependencies:**
* `tensorflow >= 2.8.0`
* `pandas`, `numpy`, `scipy`
* `opencv-python`
* `tqdm`

---

## 🗂️ Repository Structure

The repository contains the core scripts to process raw video and OpenFace outputs, fuse the data streams, and build the MSFE-CTA model:

* 📄 `facial_appearance_features.py`: Extracts and processes high-level behavioral features (Action Units, Head Pose, Eye Gaze) from OpenFace outputs, applying the 3-minute sliding window with 90% overlap.
* 📄 `Video_Features.py`: Extracts deep visual appearance representations from video frames using the pre-trained Inception-ResNet-V2 model (Global Average Pooling).
* 📄 `Data_Fusion.py`: Temporally aligns and concatenates the behavioral and visual feature streams into the final robust tensor shape expected by the model.
* 📄 `MSFE-CTA_Model.py`: The complete TensorFlow/Keras architecture, including the custom `ResidualTCNBlock`, `InceptionTCNModule`, `ChannelAttention`, and `TemporalAttention` layers.

---

## 🚀 Usage Pipeline

### 1. Behavioral Feature Extraction
First, ensure you have processed your raw videos using the [OpenFace Toolkit](https://github.com/TadasBaltrusaitis/OpenFace). Then, run the behavioral extraction script:
```bash
python facial_appearance_features.py --data_dir path/to/openface/output --meta_dir path/to/metadata
```

### 2. Deep Visual Feature Extraction
Extract the 1536-dimensional embeddings from the raw video frames:
```bash
python Video_Features.py --input_dir path/to/raw/videos --output_dir path/to/save/visual_feats
```

### 3. Data Fusion
Fuse the two modalities (Visual + Behavioral) and apply the overlapping temporal windowing (Algorithm 1):
```bash
python Data_Fusion.py --visual_dir path/to/visual_feats --behavior_dir path/to/openface/output --output_file fused_dataset.npz
```

### 4. Model Training / Inference
Import and build the MSFE-CTA model in your training script:
```python
from MSFE_CTA_Model import build_msfe_cta_model

# Default shape: 5400 frames (3 mins), 1566 combined features
model = build_msfe_cta_model(input_shape=(5400, 1566))
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'RootMeanSquaredError'])

# model.fit(X_train, y_train, ...)
```

---

## 📊 Datasets

The experiments in the paper were conducted using the official partitions of the **AVEC 2013**, **AVEC 2014**, **AVEC 2017**, and **AVEC 2019** depression datasets. 

*Note: Due to privacy restrictions and licensing, the raw datasets cannot be distributed in this repository. Researchers must request access directly from the [AVEC Challenge organizers](https://audiovisual-emotion-challenge.org/).*

---

## ✒️ Citation

If you find this code or research useful in your work, please cite our paper:

```bibtex
@article{algawwam2026multitime,
  title={Multi-time scale feature extraction and attention networks for automatic depression level prediction},
  author={Al-Gawwam, Sarmad and Zaitcev, Aleksandr and Eissa, Mohammad R. and Alshwilli, Noor and Benaissa, Mohammed},
  journal={Applied Soft Computing},
  volume={186},
  pages={114052},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.asoc.2025.114052}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
