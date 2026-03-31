# May the F1 Be With You: Classification of Viral Meme Success

Final project for Applied Machine Learning STAT-UN-3061 at Columbia University.

## Project Summary

This project studies meme virality on Reddit as a binary classification problem. We use memes collected from multiple subreddits and try to predict whether a post will be **viral**, defined as being in the top 25% of posts by upvotes. Our goal is to understand how much predictive power comes from simple metadata features, and whether adding text-based sentiment features from meme titles and OCR-extracted image text can improve performance.

At the current stage, we have built a metadata-based baseline model and a first extended version with sentiment features. The results are still preliminary, and we are continuing to improve the feature set and modeling approach.

## Dataset and Preprocessing

We start by loading the raw dataset and parsing time-related columns:

- `Created Time` is converted to datetime
- `Cake Day` is converted to datetime

We then construct several derived features used throughout exploratory analysis and modeling:

- `log_upvotes`: log-transformed upvotes
- `title_len`: number of words in the title
- `ocr_len`: number of words in OCR-extracted text
- `post_hour`: posting hour
- `post_day`: day of the week
- `account_age_days`: age of the author account at posting time
- `is_viral`: binary target, where posts above the 75th percentile of upvotes are labeled as viral

After cleaning, the dataset is split into train, validation, and test sets for modeling.

## Sentiment Feature Generation

To enrich the metadata features, we generate sentiment features from both:

- meme titles
- OCR-extracted text from meme images

We use the Hugging Face `transformers` sentiment pipeline with:

- `distilbert-base-uncased-finetuned-sst-2-english`

The model is run in batches on both text sources. For each input, we convert the output into the following features:

- `compound`: signed sentiment score, from -1 to 1
- `extreme`: absolute sentiment strength
- `is_pos`: binary indicator for positive sentiment
- `is_neg`: binary indicator for negative sentiment

These features are saved to `bert_sentiment.csv` so they can be reused without rerunning inference.

## Modeling

### Baseline
The baseline model is an **XGBoost classifier** using metadata features only.  
This serves as our initial benchmark.

### Model V1
Model V1 extends the baseline by adding **sentiment features from title and OCR text**.

## Current Status

- Baseline model completed
- Model V1 completed
- Results are promising but still preliminary
- We are still working on improving the models, tuning features, and exploring stronger multimodal approaches

## Repository Structure

- `EDA_Final.ipynb`: exploratory data analysis
- `Model Baseline.ipynb`: metadata-only XGBoost baseline
- `ModelV1.ipynb`: baseline + sentiment features
- `bert_sentiment.csv`: generated sentiment features
- `First Draft of Blog Post.pdf`: project write-up for prototype

## Next Steps

Planned improvements include:

- improving feature engineering
- reducing overfitting
- exploring alternative machine learning methods beyond the current XGBoost setup
- adding stronger text and image representations
- testing multimodal deep learning models for better virality prediction

## Notes

The current codebase is still under active development. Some file paths and directory settings are based on personal/local environments, so the project is not yet fully configured for one-click reproducibility across different machines. Users may need to adjust paths and setup details based on their own system before running the code.
