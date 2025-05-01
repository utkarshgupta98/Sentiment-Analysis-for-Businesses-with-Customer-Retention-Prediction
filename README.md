# Sentiment-Analysis-for-Businesses-with-Customer-Retention-Prediction

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
