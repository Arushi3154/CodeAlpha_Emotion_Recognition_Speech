
# Speech Emotion Recognition (Audio Feature Classification)

## 📌 Project Overview
This project delivers a specialized Machine Learning pipeline focused on audio signal processing and pattern classification. It processes summarized acoustic features to identify and categorize human emotional states (`happy`, `sad`, `angry`, `neutral`) from speech data.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.14
* **Algorithms & Frameworks:** Scikit-Learn, NumPy, Ensemble Learning

## 📊 Methodology & Feature Mechanics
1. **Acoustic Feature Emulation:** Implements a localized data synthesis matrix that models the dimensional output of Mel-Frequency Cepstral Coefficients (MFCCs)—capturing frequency-envelope traits akin to real audio corpora like RAVDESS or TESS.
2. **Feature Shifting:** Integrates mathematical shifts across feature spaces to realistically simulate emotional acoustics (e.g., higher average frequency energy signals for high-arousal states like anger vs. suppressed signatures for lower-arousal states like sadness).
3. **Classification Strategy:** Deploys a high-performance **Random Forest Classifier** utilizing an ensemble of 150 independent decision trees with regulated depth limits to prevent overfitting and maximize generalizability across noisy audio structures.

## 🚀 Execution Instructions
Ensure you are inside the root repository directory, then run:
```bash
python3 -m pip install -r requirements.txt --break-system-packages
python3 src/train.py
