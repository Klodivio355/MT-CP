
<!-- =========================
MT-CP — README TEMPLATE
Replace placeholders: <...>
========================= -->

<div align="center">

<!-- Banner / Hero image -->
<img src="model_trace_back_2" alt="MT-CP banner" width="60%" />

# Optimizing Dense Visual Predictions Through Multi-Task Coherence and Prioritization - WACV 2025 
### <Optimizing Dense Visual Predictions Through Multi-Task Coherence and Prioritization>

<!-- Badges (edit as needed) -->
<p>
  <a href="https://github.com/Klodivio355/MT-CP/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Klodivio355/MT-CP?style=for-the-badge"></a>
  <a href="https://github.com/Klodivio355/MT-CP/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/Klodivio355/MT-CP?style=for-the-badge"></a>
  <a href="https://github.com/Klodivio355/MT-CP/issues"><img alt="Issues" src="https://img.shields.io/github/issues/Klodivio355/MT-CP?style=for-the-badge"></a>
  <a href="https://github.com/Klodivio355/MT-CP/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Klodivio355/MT-CP?style=for-the-badge"></a>
</p>

</div>

---

## ✨ Overview

**MT-CP** is a fully-supervised MTL model which leverages state-of-the-art vision transformers with task-specific decoders. Our model implements a trace-back method that improves both cross-task geometric and predictive features. 
Furthermore, we present a novel dynamic task balancing approach that projects task losses onto a common scale and prioritizes more challenging tasks during training.

**Use cases**
- ✅ MTL on NYUDV2 (Semantic Segmentation, Depth Estimation and Surface Normals Estimation)
- ✅ MTL on PASCAL-Context (Semantic Segmentation, Human Parsing, and Saliency Estimation)
- ✅ A new Loss Prioritization Scheme (LPS) MTL loss which dynamically prioritizes difficult tasks during training while projecting them on a similar logarithmic scale. 

> **Paper/Report:** https://openaccess.thecvf.com/content/WACV2025/papers/Fontana_Optimizing_Dense_Visual_Predictions_Through_Multi-Task_Coherence_and_Prioritization_WACV_2025_paper.pdf

---

## 🔥 Results Visualisation

<div align="center">
  <img src="nyud_vis.pdf", width="90%" />
</div>

---

## 🚀 Replication

We follow the organisation of MTFormer (https://github.com/xiaogang00/MTFormer). Our code is validated on a NVIDIA GPU using CUDA 10.2

- **Replicate our environment:** Make sure to replicate our environment packages by running :
```
conda create --name mtcp python=3.6

git clone https://github.com/Klodivio355/MT-CP

pip install -r requirements.txt
```

- Please modify the corresponding path in "utils/mypath.py" (db_root and seism_root), "utils/common_config.py" (pretrain_path), and "configs/env.yml" (root_dir)
  
- **Download the datasets:** You can download the datasets NYUD-v2 (https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view) and PASCAL (https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz) processed by https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch.

- **Train:** You can run our experiments by running :
```
python main_CL_nyud.py --config_env configs/env.yml --config_exp configs/nyud/MultiTaskModel.yml
```
for NYUDv2. Or,
```
python main_CL_pascal.py --config_env configs/env.yml --config_exp configs/pascal/MultiTaskModel.yml
```
for PASCAL.
