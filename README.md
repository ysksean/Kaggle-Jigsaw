# 🛡️ Kaggle Jigsaw: Agile Community Rules violation Detection

## 🏆 Competition Results
- **Rank**: 123 / 2,445 (Top 6%)
- **Medal**: 🥉 Bronze Medal
- **Competition**: [Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

## 📂 Environment & Data Sources

### Input Datasets
*   `/kaggle/input/qwen2-5-32b-gptq-int4-batch4-full`
*   `/kaggle/input/qwen3-8b-embedding`
*   `/kaggle/input/huggingfacedebertav3variants`

### Pre-trained Models
*   `/kaggle/input/baai/transformers/bge-base-en-v1.5/1`
*   `/kaggle/input/baai/transformers/bge-small-en-v1.5/1`
*   `/kaggle/input/jigsaw-pretrain-public/pytorch/llama-3.2-3b-instruct/1`
*   `/kaggle/input/qwen-3-embedding/transformers/0.6b/1`
*   `/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1`

---

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

> **"대규모 언어 모델(LLM)과 다양한 NLP 기법의 앙상블을 통한 커뮤니티 규칙 위반 탐지 솔루션"**

---

## 📝 프로젝트 개요

본 프로젝트는 **Kaggle Jigsaw - Agile Community Rules** 경진대회를 위해 작성된 고성능 AI 솔루션입니다. Reddit과 같은 커뮤니티에서 사용자가 작성한 댓글이 특정 규칙을 위반했는지 여부를 자동으로 판단하는 문제를 해결합니다.

### 💡 핵심 가치 제안 (Key Value Proposition)
*   **다각도 검증 (Multi-Strategy)**: 단순 분류를 넘어 시맨틱 검색, LLM 추론, 거리 기반 학습(Triplet Loss) 등 4가지 서로 다른 전략을 결합하여 정확도를 극대화했습니다.
*   **고도의 최적화 (Efficiency)**: LoRA(Low-Rank Adaptation) 및 4-bit 양자화를 통해 제한된 GPU 리소스 내에서 대규모 모델(Qwen, Llama 3)을 효과적으로 튜닝했습니다.
*   **유연한 앙상블 (Robust Ensemble)**: 각 모델의 예측값을 가중 랭크 블렌딩(Weighted Rank Blending)하여 단일 모델의 편향을 제거하고 일반화 성능을 높였습니다.

### ✨ 주요 특징
*   **Qwen 2.5/3 LoRA Fine-tuning**: 최신 LLM을 활용한 Binary Classification.
*   **Semantic Search with Qwen Embedding**: 규칙과 댓글 간의 의미론적 유사도 측정.
*   **Triplet Loss Learning**: Sentence Transformer 기반의 거리 학습을 통한 미세 분류.
*   **DeBERTa v3 Ensemble**: 다중 시드(Multi-seed) 학습 및 랭크 평균화.

---

## 🏗️ 시스템 아키텍처

```ascii
[ Input Data ]
      |
      +-------------------------------------------------------------+
      |                                                             |
[ LLM Classifier ]      [ Semantic Search ]      [ Triplet Model ]      [ DeBERTa v3 ]
(Qwen/Llama LoRA)       (Qwen Embedding)         (BGE-base-v1.5)        (Sequence Class.)
      |                        |                        |                       |
      +----------------+-------+----------------+-------+-----------------------+
                       |                        |
             [ Rank Normalization [0, 1] ]    [ Multi-Seed Average ]
                       |                        |
                       +-----------+------------+
                                   |
                       [ Weighted Rank Blending ]
                                   |
                       [ Final Prediction (CSV) ]
```

---

## 🚀 주요 기능

- [x] **LLM 기반 이진 분류**: LoRA 기법을 활용하여 Llama 3.2 및 Qwen 모델을 규칙 위반 여부 판단(Yes/No)에 최적화.
- [x] **시맨틱 검색 (FAISS)**: 규칙 임베딩과 댓글 임베딩 간의 거리 측정을 통한 위반 가능성 산출.
- [x] **Triplet Loss 최적화**: Anchor(규칙), Positive(위반예시), Negative(정상예시)를 활용한 강건한 임베딩 공간 학습.
- [x] **고성능 앙상블**: 4가지 다른 베이스 아키텍처의 예측값을 Rank 기반으로 융합하는 블렌더 구현.

---

## 🛠️ 기술 스택

| 분류 | 기술 | 상세 이유 |
| :--- | :--- | :--- |
| **Backend/ML** | Python, PyTorch | 데이터 처리 및 모델 학습의 표준 환경 |
| **LLM/NLP** | Hugging Face Transformers | 최신 사전 학습 모델(Qwen, DeBERTa 등) 로딩 및 파인튜닝 |
| **Optimization** | PEFT (LoRA), BitsAndBytes | GPU 메모리 효율성을 위한 양자화 및 저대역폭 학습 |
| **Training** | Accelerate, DeepSpeed | 분산 학습 및 학습 속도 개선을 위한 툴킷 |
| **Similarity** | FAISS, Sentence Transformers | 고속 시맨틱 검색 및 정밀 임베딩 생성 |
| **Data** | Pandas, Scikit-learn, Datasets | 대용량 텍스트 데이터 전처리 및 분석 |

---

## 📂 프로젝트 구조

```text
.
├── fork-of-triplet-deberta-qwen-ver2-46519d.ipynb  # 메인 분석 및 실행 환경
├── constants.py       # 모델 경로, 하이퍼파라미터 등 공통 설정
├── utils.py           # 데이터 전처리, 프롬프트 생성 유틸리티
├── train.py           # LLM SFT(Supervised Fine-tuning) 스크립트
├── inference.py       # vLLM 기반 고속 추론 스크립트
├── semantic.py        # 시맨틱 검색 기반 스코어링 로직
├── triplet.py         # Triplet Loss 학습 및 클러스터링 기반 예측
├── deberta.py         # DeBERTa v3 다중 시드 학습 및 예측
└── README.md          # 프로젝트 설명 문서
```

---

## 🧠 핵심 알고리즘 및 앙상블 로직

### 1. Weighted Rank Blending
모델별로 스케일이 다른 예측값을 동일 선상에서 비교하기 위해 랭크 정규화(Rank Normalization)를 적용한 후, 각 모델의 변동성(Variance)과 중요도(Prior)에 따라 가중치를 부여합니다.

```python
# 가중 랭크 블렌딩 예시
df["rule_violation"] = (
    w_qwen * df["qwen_rank"] + 
    w_triplet * df["triplet_rank"] + 
    w_deberta * df["deberta_rank"]
)
```

### 2. Triplet Loss Learning
Anchor(규칙)와 Positive(위반 사례) 간의 거리는 좁히고, Anchor와 Negative(정상 사례) 간의 거리는 멀어지도록 학습하여 모델이 규칙의 핵심을 파악하게 합니다.

---

## ⚙️ 설치 및 실행

### 환경 요구사항
*   NVIDIA GPU (VRAM 16GB 이상 권장)
*   Python 3.10+
*   CUDA 11.8+

### 설치 가이드
1.  레포지토리 클론
    ```bash
    git clone https://github.com/ysksean/Kaggle-Jigsaw.git
    cd Kaggle-Jigsaw
    ```
2.  필수 라이브러리 설치
    ```bash
    pip install torch transformers peft accelerate bitsandbytes sentence-transformers vllm
    ```

### 실행 방법
노트북 파일을 실행하거나 개별 스크립트를 순차적으로 실행합니다.
```bash
# DeBERTa 학습 및 예측
python deberta.py

# 시맨틱/Triplet 기반 예측
python semantic.py
python triplet.py

# 최종 블렌딩은 노트북 내의 앙상블 셀 참고
```

---

## 📈 성능 최적화

*   **vLLM Inference**: LLM 추론 시 vLLM 엔진을 도입하여 기존 Transformers 대비 약 3~5배 빠른 추론 속도 달성.
*   **Gradient Checkpointing**: 대규모 모델 학습 시 메모리 부족 문제를 해결하기 위해 적용.
*   **DeepSpeed ZeRO-2**: 분산 학습 시 Optimizer State를 GPU별로 분산하여 효율적 자원 활용.

---

## 👨‍💻 개발자 정보

*   **Name**: 김유신
*   **Email**: timelesseda@gmail.com
*   **GitHub**: [github.com/ysksean](https://github.com/ysksean)

