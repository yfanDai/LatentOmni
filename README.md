<p align="center">
  <img src="assets/Logo.png" width="620" alt="LatentOmni" />
</p>

<div align="center">
  <a href="#"><img src="https://img.shields.io/badge/Project-Page-blue" height="25" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-red" height="25" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Code-Coming%20Soon-lightgrey" height="25" alt="Code"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-LatentOmni--Instruct--35K-green" height="25" alt="Dataset"></a>
</div>

## News

- **[2026.05.13]** We initialize the LatentOmni repository with the project overview, visual assets, and release roadmap.

## Todo List

We are actively preparing to release the following:

- [x] Project README and visual assets
- [ ] Paper and project page
- [ ] Training and inference code
- [ ] LatentOmni model checkpoints and evaluation scripts
- [ ] LatentOmni-Instruct-35K dataset

## LatentOmni

**LatentOmni** is a cross-modal reasoning framework for audio-visual multimodal large language models. Instead of relying only on text-based chain-of-thought, which can compress dense sensory evidence into discrete language tokens, LatentOmni performs joint reasoning in a unified continuous latent space. This design preserves rich audio-visual representations and keeps the model grounded in the original sensory inputs throughout the reasoning process.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="LatentOmni overview" />
</p>
<p align="center">
  <i>Overview of LatentOmni. The framework bridges audio, visual, and textual semantics through unified latent reasoning, feature-level supervision, and temporally synchronized audio-visual position encoding.</i>
</p>

### Abstract

While joint audio-visual understanding is fundamental to advancing machine cognition, current multimodal large language models (MLLMs) still struggle with complex cross-modal reasoning. Existing text-based chain-of-thought (CoT) compresses rich multi-modal features into discrete text, incurring information loss and inducing a language-bound phenomenon that diminishes attention to audio-visual signals. In contrast, a continuous latent space inherently preserves dense representations, serving as an ideal carrier for audio-visual information.

Motivated by this, we propose **LatentOmni**, a novel cross-modal reasoning framework. By introducing a feature-level supervision mechanism to directly reconstruct raw sensory inputs within the latent space, LatentOmni leverages native latent features to bridge audio-visual modalities and text, ensuring sustained attention on original audio-visual inputs throughout reasoning. Furthermore, to maintain temporal consistency across modalities in latent space, we design Omni-Sync Position Embedding (OSPE), which generalizes multimodal rotary position encodings to drive audio-visual synchrony.

To supervise this reasoning process, we construct **LatentOmni-Instruct-35K**, a dataset interleaving text with audio-visual segments that serve as dense evidence for latent reconstruction. Comprehensive evaluation across multiple audio-visual reasoning benchmarks demonstrates that LatentOmni substantially outperforms strong explicit-CoT baselines, validating latent space joint reasoning as a promising path toward genuine omnimodal understanding.

### Contributions

- **Unified latent audio-visual reasoning.** We propose **LatentOmni**, a novel audio-visual reasoning framework that equips MLLMs with a tailored post-training pipeline to conduct joint reasoning in a unified latent space.
- **Feature-level latent supervision and OSPE.** We introduce explicit feature-level supervision in latent space and Omni-Sync Position Embedding (OSPE) to facilitate cross-modal temporal alignment, preserve attention to audio-visual modalities, and bridge audio-visual evidence with textual semantics.
- **Audio-visual interleaved CoT data synthesis.** We develop a new audio-visual interleaved CoT data synthesis pipeline and construct **LatentOmni-Instruct-35K**, a high-quality dataset for complex cross-modal latent reasoning.
- **Strong empirical results.** Extensive experiments show that LatentOmni substantially outperforms numerous explicit-CoT models and achieves competitive performance on challenging audio-visual reasoning benchmarks.

## Method Overview

LatentOmni introduces latent reasoning tokens that carry dense audio-visual information inside the reasoning trajectory. During post-training, the model is supervised not only through text prediction, but also through reconstruction-oriented feature-level objectives that align latent states with the original audio and video evidence.

OSPE further aligns audio and visual streams in the latent space by extending multimodal rotary position embeddings to temporally synchronized audio-visual inputs. This helps the model reason over event timing, segment-level correspondence, and cross-modal causal evidence.

## LatentOmni-Instruct-35K

<p align="center">
  <img src="assets/LatentOmni-Instruct-35K.png" width="100%" alt="LatentOmni-Instruct-35K data synthesis pipeline" />
</p>
<p align="center">
  <i>LatentOmni-Instruct-35K is built through an audio-visual data synthesis and filtering pipeline that produces interleaved trajectories with dense audio-visual evidence for latent reconstruction.</i>
</p>

LatentOmni-Instruct-35K is designed to fill the training-data gap for latent-space cross-modal reasoning. The pipeline synthesizes audio-visual question-answer pairs, filters them for reasoning difficulty, logical soundness, and modality dependency, and constructs trajectories that interleave textual reasoning with aligned audio-visual segments.

## Quick Start

Code, checkpoints, evaluation scripts, and dataset download instructions are coming soon.

## Visualizations

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

## Acknowledgements

We thank the open-source multimodal learning community for building the datasets, models, and evaluation protocols that make audio-visual reasoning research possible. The README structure is inspired by the public CoF-T2I repository.

## Citation

If you find this project useful, please consider citing our work. The BibTeX entry will be updated once the paper is publicly available.

## License

The license will be updated before the full code release.
