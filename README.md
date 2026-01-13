# Offensive-Language-Detection-TriFusion

TriFusion-based **Offensive Language Detection** system that combines **CNN-GRU (word-level features)**, **CharCNN (obfuscation-aware subword features)**, and **BERT (contextual embeddings)** using a **gated fusion mechanism**.  
The model is trained and evaluated on the **Davidson dataset** with a **3-class setup**: `hate_speech`, `offensive_language`, `neither`, and includes **adversarial robustness testing** (misspellings, symbols, emojis).

---

## Key Features
- ✅ TriFusion architecture: **CNN-GRU + CharCNN + BERT**
- ✅ **Gated fusion** to dynamically weight channel importance
- ✅ Stronger **minority hate-speech detection**
- ✅ **Robustness evaluation** under adversarial text obfuscation
- ✅ Clean reproducible training/evaluation pipeline

---

## Dataset
This project uses the Davidson dataset:
- Automated Hate Speech Detection and the Problem of Offensive Language (Davidson et al., ICWSM 2017)

⚠️ Dataset is not included in this repo. Download it separately and place it as:

