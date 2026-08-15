# Bilingual Financial Complaint Classification System 
*A Comparative Study of Machine Learning and Transformer Models for English–Hindi Complaint Analysis*

---

## Overview
This project presents an intelligent **bilingual financial complaint classification system** designed to automate the categorization of consumer complaints written in **English and Hindi**.  
It enables organizations to analyze and route customer complaints more efficiently, minimizing manual workload and improving the fairness and speed of grievance redressal.

The system compares **traditional machine learning models** (Logistic Regression, LightGBM, Naive Bayes, Random Forest) with a **fine-tuned multilingual transformer model (XLM-RoBERTa)** to identify the most effective approach for bilingual NLP.

---

## Tech Stack
| Category | Tools & Libraries |
|----------|-------------------|
| **Language** | Python 3.11 |
| **NLP / Transformers** | XLM-RoBERTa, Hugging Face Transformers, NLTK |
| **Machine Learning** | Scikit-learn, LightGBM |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Translation** | Helsinki-NLP/opus-mt-en-hi |
| **Environment** | Jupyter Notebook, Kaggle GPU |

---

## Project Structure
```
FinancialComplaintClassification/
├── notebooks/
│   ├── nlp-project-dataset-building-50k-samples.ipynb   # Dataset construction & translation
│   ├── 50k-samples-baseline-models-with-tf-idf.ipynb    # Baseline ML models with TF-IDF
│   ├── ensemble-model.ipynb                              # Voting ensemble approach
│   └── Final_NLP_Project_Fine_Tuning.ipynb               # XLM-RoBERTa fine-tuning
├── src/
│   ├── config.py                    # Centralized path configuration
│   ├── dataset_analysis_cleaning.py # Raw dataset analysis & cleaning
│   ├── eda.py                       # Exploratory data analysis
│   └── preprocessing.py            # Text preprocessing pipeline
├── sample_bilingual_complaints.csv  # Sample bilingual dataset
├── .gitignore
└── README.md
```

---

## Objectives
- Develop a bilingual complaint categorization system supporting both English and Hindi text.  
- Enable multi-class classification of financial complaint categories.  
- Automate complaint routing to reduce manual triage time.  
- Identify and prioritize critical complaints for faster redressal.  
- Deliver a deployable prototype demonstrating real-world usability.

---

## Dataset
**Source:** [Consumer Financial Protection Bureau (CFPB) Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)  
This open dataset contains millions of anonymized financial complaints.

**Final Dataset Summary**
- **Size:** 25,000 balanced samples (5,000 per top 5 complaint categories)  
- **Languages:** English and Hindi  
- **Preparation Process:**  
  1. Cleaned and filtered complaints with complete narratives.  
  2. Stratified sampling to ensure class balance.  
  3. English-to-Hindi translation using *Helsinki-NLP/opus-mt-en-hi*.  
  4. Language-specific preprocessing: stopword removal, lemmatization, and Devanagari character filtering.

---

## Methodology

### Data Preprocessing
A multi-step pipeline was developed:
1. **Cleaning and Normalization:** Removed missing narratives, lowercased text, stripped punctuation.  
2. **Balancing:** Applied stratified sampling across top complaint categories.  
3. **Bilingual Corpus Creation:** Generated parallel English–Hindi dataset.  
4. **Advanced Text Cleaning:** Removed noise and normalized text for both languages.

### Feature Engineering
- **TF-IDF Word-Level Features:** Captured term importance for English text.  
- **Character-Level TF-IDF (Hindi):** Captured morphological structure of Hindi text.  
- **Stacked Bilingual Features:** Combined representations from both languages.  
- **Transformer Tokenization:** Prepared bilingual text for transformer fine-tuning.

### Models Compared
| Model | Type | Accuracy |
|--------|-------|-----------| 
| Logistic Regression | Traditional ML | 86.62% |
| Multinomial Naive Bayes | Traditional ML | 83.45% |
| Random Forest | Ensemble ML | 84.42% |
| LightGBM | Gradient Boosting | 87.18% |
| Voting Ensemble | Hybrid ML | 87.75% |
| **XLM-RoBERTa (Fine-Tuned)** | **Multilingual Transformer** | **88.38%** |

### Evaluation Metrics
Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

The bilingual dataset was split 80:20 for training and testing, ensuring balanced evaluation.

---

## Results and Discussion
- **LightGBM** provided the strongest traditional baseline at 87.18% accuracy.  
- **Voting Ensemble** slightly improved results to 87.75%.  
- **Fine-tuned XLM-RoBERTa** achieved the highest accuracy of **88.38%**, confirming the advantage of multilingual transformers for bilingual tasks.  

This indicates that contextual deep learning models outperform frequency-based approaches in understanding semantic nuances across languages.

---

## Impact
- **Automation of Complaint Handling:** Reduces human dependency and operational delays in grievance redressal.  
- **Scalable Bilingual NLP Solution:** Efficiently processes English and Hindi consumer complaints.  
- **Fair and Objective Classification:** Reduces bias by applying data-driven analysis.  
- **Faster Customer Response:** Prioritizes urgent or severe complaints automatically.  
- **Foundation for Multilingual Systems:** Establishes a scalable architecture for future multilingual complaint management in diverse linguistic contexts.

---

## Conclusion
The project demonstrates that **fine-tuned multilingual transformers**, particularly XLM-RoBERTa, deliver superior accuracy and contextual understanding compared to traditional ML methods.  
It establishes a strong foundation for real-world deployment of bilingual NLP systems in customer service automation.

---

## Author
**Ramsaheb Prasad**
