# MultiMAE-DER: Multimodal Masked Autoencoder for Dynamic Emotion Recognition (IEEE ICPRS 2024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/emotion-recognition-on-ravdess)](https://paperswithcode.com/sota/emotion-recognition-on-ravdess?p=multimae-der-multimodal-masked-autoencoder)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/video-emotion-recognition-on-crema-d)](https://paperswithcode.com/sota/video-emotion-recognition-on-crema-d?p=multimae-der-multimodal-masked-autoencoder)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimae-der-multimodal-masked-autoencoder/multimodal-emotion-recognition-on-iemocap)](https://paperswithcode.com/sota/multimodal-emotion-recognition-on-iemocap?p=multimae-der-multimodal-masked-autoencoder)<br>

> [`Website`](https://hcps.fiu.edu/) | [`arXiv`](https://arxiv.org/abs/2404.18327) | [`BibTeX`](#citation)<br>
> [Peihao Xiang](https://scholar.google.com/citations?user=k--3fM4AAAAJ&hl=zh-CN&oi=ao), [Chaohao Lin](https://scholar.google.com/citations?hl=zh-CN&user=V3l7dAEAAAAJ), [Kaida Wu](https://ieeexplore.ieee.org/author/167739911238744), and [Ou Bai](https://scholar.google.com/citations?hl=zh-CN&user=S0j4DOoAAAAJ)<br>
> HCPS Laboratory, Department of Electrical and Computer Engineering, Florida International University<br>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Peihao-Xiang/MultiMAE-DER/blob/main/MultiMAE-DER_Fine-Tuning%20Code/MultiMAE_DER_FSLF.ipynb)  [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/NoahMartinezXiang/RAVDESS)

Official TensorFlow implementation and pre-trained VideoMAE models for MultiMAE-DER: Multimodal Masked Autoencoder for Dynamic Emotion Recognition.

## Overview

<p align="center">
  <img src="images/MultiMAE-DER.png" width=50%
    class="center"><br>
  Illustration of our MultiMAE-DER.
</p>

General Multimodal Model vs. MultiMAE-DER. The uniqueness of our approach lies in the capability to extract features from cross-domain data using only a single encoder, eliminating the need for targeted feature extraction for different modalities.

<p align="center">
  <img src="images/Multimodal_Sequence_Fusion_Strategy.png" width=70%
    class="center"><br>
  Multimodal Sequence Fusion Strategies.
</p>

## Implementation details

<p align="center">
  <img src="images/MultiMAE-DER_Program_Flowchart.png" width=50%> <br>
  The architecture of MultiMAE-DER.
</p>

## Main Results

### RAVDESS

![Result_on_RAVDESS](images/Result_on_RAVDESS.png)

### CREMA-D

![Result_on_CREMA-D](images/Result_on_CREMA-D.png)

### IEMOCAP

![Result_on_IEMOCAP](images/Result_on_IEMOCAP.png)

## Contact 

If you have any questions, please feel free to reach me out at pxian001@fiu.edu.

## Acknowledgments
This project is built upon [VideoMAE](https://github.com/innat/VideoMAE) and [MAE-DFER](https://github.com/sunlicai/MAE-DFER). Thanks for their great codebase.

## License

This project is under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you find this repository helpful, please consider citing our work:

```BibTeX
@misc{xiang2024multimaeder,
      title={MultiMAE-DER: Multimodal Masked Autoencoder for Dynamic Emotion Recognition}, 
      author={Peihao Xiang and Chaohao Lin and Kaida Wu and Ou Bai},
      year={2024},
      eprint={2404.18327},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@INPROCEEDINGS{10677820,
  author={Xiang, Peihao and Lin, Chaohao and Wu, Kaida and Bai, Ou},
  booktitle={2024 14th International Conference on Pattern Recognition Systems (ICPRS)}, 
  title={MultiMAE-DER: Multimodal Masked Autoencoder for Dynamic Emotion Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-7},
  keywords={Emotion recognition;Visualization;Correlation;Supervised learning;Semantics;Self-supervised learning;Transformers;Dynamic Emotion Recognition;Multimodal Model;Self-Supervised Learning;Video Masked Autoencoder;Vision Transformer},
  doi={10.1109/ICPRS62101.2024.10677820}}
```
