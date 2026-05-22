<p align="center">
  <img src="assets/Logo.png" width="620" alt="LatentOmni" />
</p>

<div align="center">
<p style="margin-top:10px;">
  <a href="https://arxiv.org/abs/2605.22012"><img src="https://img.shields.io/badge/arXiv-2605.22012-red?style=badge&logo=arXiv" alt="arXiv" height="25"></a>
  <a href="https://huggingface.co/papers/2605.22012"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-2605.22012-blue" height="25"></a>
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Coming%20Soon-lightgrey" height="25"></a>
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Coming%20Soon-lightgrey" height="25"></a>
  <a href="https://github.com/yfanDai/LatentOmni"><img src="https://img.shields.io/github/stars/yfanDai/LatentOmni?style=social" alt="GitHub Stars" height="25"></a>
</p>
</div>

## 🔥 News

- **[2026.05.22]** We initialize the LatentOmni repository with the project overview, visual assets, and release roadmap.

## 🎯 Todo List

We are actively preparing to release the following:

- [x] Project README and Paper
- [ ] Training and inference code
- [ ] LatentOmni model checkpoints
- [ ] LatentOmni-Instruct-35K dataset

## 🎬 LatentOmni

**LatentOmni** is a cross-modal reasoning framework for audio-visual multimodal large language models. Instead of relying only on text-based chain-of-thought,  LatentOmni performs joint reasoning in a unified continuous latent space. This design preserves rich audio-visual representations and keeps the model grounded in the original sensory inputs throughout the reasoning process.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="LatentOmni overview" />
</p>
<p align="center">
  <i>LatentOmni Overview. Left: A start embedding automatically triggers latent reasoning. The model autoregressively generates latent audio-visual embeddings, temporally synchronized via OSPE for precise cross-modal reasoning. Right: Training employs three specialized losses and a controlled attention flow, explicitly guiding the model to leverage latent representations for downstream reasoning.</i>
</p>

## 🔍 Contributions

- **Unified latent audio-visual reasoning.** We propose **LatentOmni**, enabling joint reasoning in a unified latent space for MLLMs.
- **Feature-level latent supervision and OSPE.** We introduce latent reconstruction supervision and Omni-Sync Position Embedding to align cross-modal temporal dynamics and preserve audio-visual attention.
- **Audio-visual interleaved CoT data synthesis.** We construct **LatentOmni-Instruct-35K**, a high-quality dataset with audio-visual interleaved reasoning trajectories.
- **Strong empirical results.** LatentOmni substantially outperforms explicit-CoT baselines on challenging audio-visual reasoning benchmarks.

## 📑 LatentOmni-Instruct-35K
LatentOmni-Instruct-35K is designed to fill the training-data gap for latent-space cross-modal reasoning. The pipeline synthesizes audio-visual question-answer pairs, filters them for reasoning difficulty, logical soundness, and modality dependency, and constructs trajectories that interleave textual reasoning with audio-visual segments.

<p align="center">
  <img src="assets/LatentOmni-Instruct-35K.png" width="100%" alt="LatentOmni-Instruct-35K data synthesis pipeline" />
</p>
<p align="center">
  <i>LatentOmni-Instruct-35K Dataset Construction Pipeline, including AVQA Data Synthesis & Filtering, Segment-Level Caption Synthesis, and AV-Interleaved Reasoning Trajectory Synthesis.</i>
</p>


## 🚀 Quick Start

Code, checkpoint, and dataset preprocess instructions are coming soon.

## 📊 Results Analysis
### 1️⃣ Performance comparison on Omni Understanding Benchmarks

<p align="center">
  <img src="assets/Table1.png" width="100%" alt="LatentOmni reasoning visualization" />
</p>
<p align="center">
  <i>Performance comparison of proprietary and open-source models on Daily-Omni, WorldSense, OmniVideoBench, and LVOmniBench benchmarks. The <b>best</b> is highlighted.</i>
</p>

### 2️⃣ Fine-Grained Analysis on OmniVideoBench

<p align="center">
  <img src="assets/Table2.png" width="100%" alt="LatentOmni reasoning visualization" />
</p>
<p align="center">
  <i>Accuracy comparison of LatentOmni and other methods on OmniVideoBench. The <b>best</b> is highlighted and the second-best is <u>underlined</u>. Performance gains over the base model are shown in red parentheses.</i>
</p>

### 3️⃣ Performance comparison with other Latent Reasoning Methods on VideoMME

<p align="center">
  <img src="assets/Table3.png" width="100%" alt="LatentOmni reasoning visualization" />
</p>
<p align="center">
  <i>Performance comparison with recent visual latent reasoning methods on the VideoMME benchmark (without audio inputs). The <b>best</b> is highlighted.</i>
</p>

## 🔭 Visualizations

<details open>
  <summary><i>Click to collapse or expand visual examples.</i></summary>

### Audio-Visual Reasoning Comparison

<p align="center">
  <img src="assets/visualization.png" width="100%" alt="LatentOmni reasoning visualization" />
</p>
<p align="center">
  <i>LatentOmni maintains stronger attention to audio-visual evidence than explicit text-based CoT baselines and improves reasoning over temporally grounded cross-modal clues.</i>
</p>

### Example: Audio-Visual Event Alignment

<p align="center">
  <img src="assets/case1.png" width="92%" alt="LatentOmni audio-visual event alignment example" />
</p>
<p align="center">
  <i>A representative DailyOmni example showing how LatentOmni identifies temporally aligned audio-visual evidence before producing the final answer.</i>
</p>

</details>

## 📖 Citation

If you find this project useful, please consider citing our work. The BibTeX entry will be updated once the paper is publicly available.


## 📒 License

This repository is released under the MIT License. See [LICENSE](LICENSE) for additional details.
