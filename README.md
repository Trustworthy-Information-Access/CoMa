# CoMa: Compression then Matching: An Efficient Pre-training Paradigm for Multimodal Embedding

<p align="center">
        &nbsp&nbsp 📙 <a href="https://arxiv.org/pdf/2511.08480">Our Paper</a>&nbsp&nbsp |
        &nbsp&nbsp 🤗 <a href="https://huggingface.co/404-not-founds/CoMa">Our Models</a>&nbsp&nbsp
</p>

## Introduction
An effective embedding is expected to comprehensively preserve the semantic content of the input while simultaneously emphasizing features that are discriminative for downstream tasks. Recent approaches demonstrate that MLLMs can be adapted into competitive embedding models via large-scale contrastive learning, enabling the simultaneous optimization of two complementary objectives. We argue that the two aforementioned objectives can be decoupled: a comprehensive understanding of the input facilitates the embedding model in achieving superior performance in downstream tasks via contrastive learning. we propose CoMa, a compressed pre-training phase, which serves as a warm-up stage for contrastive learning. Experiments demonstrate that with only a small amount of pre-training data, we can transform a MLLM into a competitive embedding model. CoMa achieves new state-of-the-art results among MLLMs of comparable size on the MMEB, realizing optimization in both efficiency and effectiveness.

## Architecture
The overall architecture is shown as follows:
<div style="display: flex;">
  <img src="doc/images/architecture.png" alt="opencompass" style="width: 100%; height: auto;" />
</div>

## Performance
<p align="center" style="display:flex;">
    <img src="./doc/images/performance.png"/>
<p>

## Data Preparation
Our pre-training data construction files are located in the data_prepare folder. To obtain the final pre-trained data, you must sequentially execute sample.py, auto_q.py, and auto_a.py. Please note that you must replace the file paths and model names within each script. We use [vllm](https://github.com/vllm-project/vllm) to deploy and invoke MLLMs.

## Environmental Requirements
The packages we use during training and inference can be found in requirements.txt
``` bash
pip install -r requirements.txt
```

## Training 
``` bash
pip install -r requirements.txt
```
## Inference 
``` bash
pip install -r requirements.txt
```

## Acknowledgements
This project is based on the work of [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec).  


## License Agreement
All of our open-source models are licensed under the [Apache-2.0](./LICENSE) license.

## We are Hiring 🔥🔥🔥
The Kuaishou-Multimodal Understanding Team focuses on multimodal large models tailored for short videos, live streaming, search recommendations, and e-commerce. We provide foundational model technology support for Kuaishou's diverse business operations. We welcome inquiries and look forward to working on challenging projects with talented individuals like you!

Location: Beijing

Contact & Resume Submission: yuanwei05@kuaishou.com, wangyan33@kuaishou.com, yangbiao@kuaishou.com

> Kuaishou多模态内容理解团队专注于适合短视频、直播、搜索推荐、电商的多模态大模型，为快手的各项业务提供基座模型技术支持，欢迎咨询(实习/全职)，期待和优秀的你，一起做有挑战的事情！
>
> 岗位城市：北京
> 
> 咨询&简历投递：yuanwei05@kuaishou.com, wangyan33@kuaishou.com, yangbiao@kuaishou.com

## Citation
```
@misc{li2025compressionmatchingefficientpretraining,
      title={Compression then Matching: An Efficient Pre-training Paradigm for Multimodal Embedding}, 
      author={Da Li and Yuxiao Luo and Keping Bi and Jiafeng Guo and Wei Yuan and Biao Yang and Yan Wang and Fan Yang and Tingting Gao and Guorui Zhou},
      year={2025},
      eprint={2511.08480},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.08480}, 
}
```

