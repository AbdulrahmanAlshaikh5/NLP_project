# Smarter Spam Detection

**Course:** ARTI 405 - Natural Language Processing (Section 7MA1)  
**Institution:** Imam Abdulrahman Bin Faisal University  
**Instructor:** Dr. Ahmed Talal  
**Group:** Group 1   

**Team Members:**
- Abdulrahman Alshaikh (2240004223) - Team Leader
- Abdulrahman Fahad AlHumam (2240002411)
- Ibrahim Khaled Alkhamis (2240004075)

---

## Project Overview

Spam filters are commonly judged on how much spam they catch, yet wrongly flagging a legitimate bank alert, OTP verification code, delivery update, or clinic reminder is often much more disruptive for users. Because legitimate automated alerts frequently share keywords with spam (such as `urgent`, `verify`, `code`, `free`, `account`, and `click`), simple filters can trigger high false-alarm rates.

**Research Question:**  
*Can a text classifier keep spam detection high while reducing how often legitimate, suspicious-looking messages are wrongly flagged as spam?*

To answer this question, we compare two traditional lexical baselines (**TF-IDF + Multinomial Naive Bayes** and **TF-IDF + Logistic Regression**) against a fine-tuned pretrained language model (**BERT**, `bert-base-uncased`) using identical leak-free splits and development-set threshold tuning.

---

## Repository Structure

```text
NLP_project/
├── Smarter_Spam_Detection.ipynb   # Main Google Colab notebook (with saved outputs)
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
└── data/
    ├── hard_cases.csv             # 100 legitimate look-alike SMS messages (25 per category)
    └── student_examples.csv       # 6 original labelled examples (2 per team member)
```

> The raw **UCI SMS Spam Collection** (`SMSSpamCollection`) is automatically downloaded and extracted into `data/` when running the notebook, and all evaluation tables and plots are generated into `results/`.

---

## How to Run (Google Colab)

1. Open `Smarter_Spam_Detection.ipynb` in **Google Colab**.
2. Enable GPU acceleration: **Runtime -> Change runtime type -> T4 GPU**.
3. Upload the `data/` folder (`hard_cases.csv` and `student_examples.csv`) into the Colab working directory.
4. Run all cells in order: **Runtime -> Restart session and run all**.
5. Use the **Try it** section at the bottom of the notebook (`predict_sms(...)`) to test any new SMS message without retraining.

---

## Summary of Results

### 1. UCI Held-Out Test Set (1,035 Messages: 911 Ham, 124 Spam)

| Model | Tuned Threshold (Dev) | Accuracy | Macro F1 | Spam Recall | Ham FPR | FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF Naive Bayes | 0.50 | 98.36% | 0.9592 | 87.90% | 0.22% | 2 / 15 |
| TF-IDF Logistic Regression | 0.50 | 98.36% | 0.9612 | 93.55% | 0.99% | 9 / 8 |
| **Fine-Tuned BERT** | **0.95** | **99.13%** | **0.9793** | **95.97%** | **0.44%** | **4 / 5** |

### 2. Hard-Case Legitimate Test Set (100 Modern Alerts: 25 OTP, 25 Bank, 25 Delivery, 25 Appointment)

| Model | Threshold | False Alarms (out of 100) | Appointment | Bank Alert | Delivery | OTP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF Naive Bayes | 0.50 | 41 (41%) | 4 | 12 | 9 | 16 |
| **TF-IDF Logistic Regression** | **0.50** | **36 (36%)** | **4** | **10** | **10** | **12** |
| Fine-Tuned BERT | 0.95 | 92 (92%) | 20 | 25 | 22 | 25 |

### Key Takeaway
While fine-tuned BERT clearly leads on the standard 2011 UCI benchmark (cutting total errors from 17 to 9), temporal domain shift between casual 2011 UCI ham and modern automated alerts, combined with overconfident softmax probabilities (>0.99), caused BERT to flag 92% of legitimate modern alerts even at threshold 0.95. Expanding training data with modern transactional ham and applying post-hoc calibration (e.g., temperature scaling) is the most practical next step.

---

## References

- T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng '11. https://doi.org/10.24432/C5CC84
- J. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019. https://aclanthology.org/N19-1423/
- Hugging Face Transformers (`bert-base-uncased`): https://huggingface.co/docs/transformers/model_doc/bert
- Scikit-learn Decision Threshold Tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- C. Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.
