# Sentiment Analysis and Customer Retention Prediction using Yelp Reviews

This project analyzes customer reviews from the Yelp Open Dataset to perform sentiment classification and generate retention insights for businesses. It integrates data engineering, natural language processing, and machine learning into an end-to-end pipeline with scalable visualization support.

---

## Dataset

- Source: [Yelp Open Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- Components used:
  - `yelp_academic_dataset_review.json`
  - `yelp_academic_dataset_business.json`
  - `yelp_academic_dataset_user.json`
  - `yelp_academic_dataset_checkin.json`
  - `yelp_academic_dataset_tip.json`
- Sample size: 100,000 entries per file

---

## Setup Instructions

1. Upload your `kaggle.json` to the working directory.
2. Run the initial setup to configure Kaggle and download the dataset:

```bash
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d yelp-dataset/yelp-dataset
!unzip yelp-dataset.zip -d yelp_dataset
```

3. Install required packages:

```bash
pip install pandas nltk seaborn matplotlib wordcloud scikit-learn mlxtend
```

---

## Modules Overview

### Module 1: Data Ingestion & Cleaning

- Load JSON datasets
- Drop duplicates and missing values
- Normalize and clean review text using:
  - Lowercasing
  - Regex punctuation removal
  - Stopword removal and lemmatization via NLTK

### Module 2: Data Integration

- Merge review data with:
  - Business metadata
  - User stats (average rating, review count)
  - Check-in and tip counts
- Output: `preprocessed_data.csv`

### Module 3: Exploratory Data Analysis (EDA)

- Distribution of ratings and categories
- Sentiment assignment:  
  - 1–2 stars → Negative  
  - 3 stars → Neutral  
  - 4–5 stars → Positive
- WordCloud visualizations
- Check-in vs. star rating analysis
- Output: `eda_ready_data.csv`

### Module 4: Sentiment Classification

- Feature: `cleaned_text`
- Label: `sentiment`
- Model: TF-IDF + Multinomial Logistic Regression
- Accuracy: ~84%
- Exported:
  - `final_sentiment_model.pkl`
  - `tfidf_vectorizer.pkl`

### Module 5: Interactive Prediction

- Function to predict sentiment of any review string
- Loads `.pkl` model and vectorizer
- Can be integrated into Streamlit UI

### Module 6: Pattern Mining & Similarity Graphs

- FP-Growth:
  - Frequent itemsets and association rules
  - Output: `freq_itemsets_final.csv`, `assoc_rules_csv_final.csv`
- Jaccard Similarity:
  - Token-based word pairs
  - Output: `jaccard_similarity_top.csv`
- MinHash LSH:
  - Approximate similarity via Hamming distance
  - Output: `minhash_lsh_approx_similarity.csv`
- Co-Occurrence Graph:
  - High-frequency word pairings (≥1500 times)
  - Output: `cooccurrence_graph_edges.csv`

---

## Key Outputs

| File | Description |
|------|-------------|
| `preprocessed_data.csv` | Cleaned and merged dataset |
| `eda_ready_data.csv` | Post-EDA dataset with sentiment |
| `final_sentiment_model.pkl` | Trained logistic regression model |
| `tfidf_vectorizer.pkl` | Vectorizer used in training |
| `freq_itemsets_final.csv` | Frequent word patterns |
| `assoc_rules_csv_final.csv` | Association rules from reviews |
| `jaccard_similarity_top.csv` | Similar word pairs (Jaccard) |
| `minhash_lsh_approx_similarity.csv` | Approximate review similarities |
| `cooccurrence_graph_edges.csv` | Word co-occurrence graph edges |

---

## Visualizations

- Sentiment distribution bar chart  
- Rating vs. check-in count line plot  
- Top business categories bar plot  
- Word clouds for positive/negative reviews  
- Correlation heatmaps  
- Graph data for Streamlit integration

---

## Deployment (Optional)

You can use Streamlit and Hugging Face Spaces to deploy:

```bash
streamlit run app.py
```

Make sure your app loads:
- `final_sentiment_model.pkl`
- `tfidf_vectorizer.pkl`
- Visualization CSVs

---

## Citation

This repository is part of an M.Tech project titled:  
**Sentiment Analysis for Businesses with Customer Retention Prediction**  
by Utkarsh Gupta (M24DE2037), IIT Jodhpur

