import pandas as pd
import numpy as np
from pathlib import Path
import sqlalchemy as sa
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_raw_data():
	"""Load raw airline incidents dataset"""
	data_dir = Path("../data")
	csv_files = list(data_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError("No CSV files found in data directory.")

	df = pd.read_csv(csv_files[0])
	print(f"Loaded {len(df)} records from {csv_files[0].name}")
	return df

def preprocess_text(text):
	"""Clean and preprocess text data"""
	if pd.isna(text):
		return ""

	# Covert to lowercase
	text = str(text).lower()
	# Remove special characters, keep spaces
	text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
	# Remove extra whitespace
	text = re.sub(r'\s+', ' ', text).strip()

	return text

def extract_text_features(df):
	"""Extract features from Report column"""
	print("Processing Report column...")

	# Clean the text
	df['report_clean'] = df['Report'].apply(preprocess_text)

	# Create TF-IDF features for individual tokens
	tfidf_vectorizer = TfidfVectorizer(
		max_features=1000,	# Top 1000 features
		stop_words='english',
		ngram_range=(1, 1), 	# Individual words
		min_df=2	# Must appear in at least 2 documents
	)

	tfidf_features = tfidf_vectorizer.fit_transform(df['report_clean'])

	# Create TF-IDF features for phrases (bigrams/trigrams)
	print("Creating phrase features...")
	phrase_vectorizer = TfidfVectorizer(
		max_features=500,	# Top 500 features
		stop_words='english',
		ngram_range=(2, 3),	# Bigrams, trigrams
		min_df=2
	)

	phrase_features = phrase_vectorizer.fit_transform(df['report_clean'])

	return tfidf_features, phrase_features, tfidf_vectorizer, phrase_vectorizer

def clean_data(df):
	"""Initial data cleaning"""
	print(f"Original shape: {df.shape}")

	# Remove completely empty rows
	df = df.dropna(how='all')
	# Basic info about missing values
	print("Missing values per column:")
	print(df.isnull().sum())

	print(f"Cleaned shape: {df.shape}")
	return df

def feature_engineering(df):
	"""Main feature engineering pipeline"""
	# Clean data
	df = clean_data(df)

	# Extract text features
	tfidf_features, phrase_features, tfidf_vec, phrase_vec = extract_text_features(df)

	# Prepare target variable (Part Failure)
	# Remove rows where Part Failure is missing
	df_ml = df[df['Part Failure'].notna()].copy()

	# Get corresponding feature matrices
	valid_indices = df_ml.index
	tfidf_ml = tfidf_features[valid_indices]
	phrase_ml = phrase_features[valid_indices]

	print(f"ML dataset shape: {df_ml.shape}")
	print(f"TF-IDF features shape: {tfidf_ml.shape}")
	print(f"Phrase features shape: {phrase_ml.shape}")

	return df_ml, tfidf_ml, phrase_ml, tfidf_vec, phrase_vec

def initial_ml_clustering(df_ml, tfidf_features, phrase_features, target_col='Part Failure'):
	"""Run initial ML to see token/phrase relationships with Part Failures"""

	print(f"\nTarget variable distribution:")
	print(df_ml[target_col].value_counts().head(10))

	# For now, focus on common failure types to ID patterns
	top_failures = df_ml[target_col].value_counts().head(20).index
	df_filtered = df_ml[df_ml[target_col].isin(top_failures)].copy()

	# Get corresponding feature matrices
	valid_mask = df_ml[target_col].isin(top_failures)
	X_tfidf = tfidf_features[valid_mask.values]
	X_phrases = phrase_features[valid_mask.values]
	y = df_filtered[target_col]

	print(f"Filtered dataset for ML: {len(df_filtered)} records")
	print(f"Number of failure types: {y.nunique()}")

	# Split data
	X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
		X_tfidf, y, test_size=0.2, random_state=42, stratify=y
	)

	# Train all models
	model_tfidf, _, _, _, _ = train_tfidf_model(X_tfidf, y)
	model_phrases, _, _, _, _ = train_phrase_model(X_phrases, y)
	model_combined, _, _, _, _ = train_combined_model(X_tfidf, X_phrases, y)

	# Create combined features for meta-learner
	from scipy.sparse import hstack
	X_combined = hstack([X_tfidf, X_phrases])

	# Train meta-learner
	print("="*50)
	print("TRAINING META-LEARNER")
	print("="*50)
	meta_model = train_meta_learner(model_tfidf, model_phrases, model_combined,
									X_tfidf, X_phrases, X_combined, y)

	return model_tfidf, model_phrases, model_combined, meta_model, df_filtered, X_tfidf, X_phrases

def train_tfidf_model(X_tfidf, y):
	"""Train model using TF-IDF features"""
	X_train, X_test, y_train, y_test = train_test_split(
		X_tfidf, y, test_size=0.2, random_state=42, stratify=y
	)

	rf = RandomForestClassifier(n_estimators=100, random_state=42)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)

	print("="*50)
	print("TF-IDF FEATURES (Individual Tokens)")
	print("="*50)
	print(classification_report(y_test, y_pred))

	return rf, X_train, X_test, y_train, y_test

def train_phrase_model(X_phrases, y):
	"""Train model using phrase features (bigrams/trigrams)"""
	X_train, X_test, y_train, y_test = train_test_split(
		X_phrases, y, test_size=0.2, random_state=42, stratify=y
	)

	rf = RandomForestClassifier(n_estimators=100, random_state=42)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)

	print("=" * 50)
	print("PHRASE FEATURES (Bigrams/Trigrams")
	print("=" * 50)
	print(classification_report(y_test, y_pred))

	return rf, X_train, X_test, y_train, y_test

def train_combined_model(X_tfidf, X_phrases, y):
	"""Train model using combined TF-IDF + Phrase Features"""
	from scipy.sparse import hstack

	X_combined = hstack([X_tfidf, X_phrases])
	X_train, X_test, y_train, y_test = train_test_split(
		X_combined, y, test_size=0.2, random_state=42, stratify=y
	)

	rf = RandomForestClassifier(n_estimators=100, random_state=42)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)

	print("=" * 50)
	print("COMBINED FEATURES (TF-IDF + Phrases")
	print("=" * 50)
	print(classification_report(y_test, y_pred))

	return rf, X_train, X_test, y_train, y_test

def train_meta_learner(model_tfidf, model_phrases, model_combined, X_tfidf, X_phrases, X_combined, y):
	"""Train a meta-model on the predictions of the three base models"""

	# Get out-of-fold predictions to avoid overfitting
	pred_tfidf = cross_val_predict(model_tfidf, X_tfidf, y, cv=5, method='predict_proba')
	pred_phrases = cross_val_predict(model_phrases, X_phrases, y, cv=5, method='predict_proba')
	pred_combined = cross_val_predict(model_combined, X_combined, y, cv=5, method='predict_proba')

	# Stack the predictions as features for meta-learner
	meta_features = np.hstack([pred_tfidf, pred_phrases, pred_combined])

	# Train meta-learner (LogReg, RF, etc.)
	meta_model = LogisticRegression()
	meta_model.fit(meta_features, y)

	# Evaluate the meta-learner using cross-validation
	cv_scores = cross_val_score(meta_model, meta_features, y, cv=5, scoring='accuracy')

	print("=" * 50)
	print(f"Meta-learner cross-validation accuracy:"
		  f"Mean: {cv_scores.mean():.4f}"
		  f"Median: {np.median(cv_scores):.4f}"
		  f"Std: +/- {cv_scores.std() * 2:.4f})")
	print("=" * 50)
	print("META-LEARNER PERFORMANCE")
	print("=" * 50)

	# Get predictions for classification report
	meta_predictions = cross_val_predict(meta_model, meta_features, y, cv=5)
	print(classification_report(y, meta_predictions))

	return meta_model

if __name__ == "__main__":
	# Load and process
	df = load_raw_data()
	df_ml, tfidf_features, phrase_features, tfidf_vec, phrase_vec = feature_engineering(df)

	# Run initial clustering/classification
	model_tfidf, model_phrases, model_combined, meta_model, df_filtered, X_tfidf, X_phrases = initial_ml_clustering(
		df_ml, tfidf_features, phrase_features
	)

	print("Initial processing complete!")
	print("All models trained including meta-learner!")

	# feature_names = tfidf_vec.get_feature_names_out()
	# importances = model_tfidf.feature_importances_
	# top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
	# print("Top 20 predictive tokens:")
	# for token, importance in top_features:
	# 	print(f"{token}: {importance:.4f}")
