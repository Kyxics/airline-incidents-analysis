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
from sklearn.metrics import classification_report, confusion_matrix

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

	return model_tfidf, model_phrases, model_combined, meta_model, df_filtered, X_tfidf, X_phrases, y

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

def analyze_vocabulary_consistency(df_filtered, y, feature_names, X_tfidf):
	"""Analyse vocabulary consistency within each fialure category"""
	from collections import defaultdict

	print(f"\nVOCABULARY CONSISTENCY ANALYSIS:")
	print("Category | Unique Tokens | Avg Tokens/Report | Consistency Score")
	print("-"*50)

	consistency_scores = []

	# For each failure category
	for category in sorted(y.unique()):
		category_mask = (y == category)
		category_reports = df_filtered[category_mask]['report_clean']
		category_tfidf = X_tfidf[category_mask.values]

		# Calculate vocabulary diversity metrics
		total_reports = len(category_reports)

		# Get non-zero features for this category
		category_features = category_tfidf.sum(axis=0).A1	# Convert to 1D array
		active_features = np.where(category_features > 0)[0]
		unique_tokens = len(active_features)

		# average tokens per report
		avg_tokens_per_report = category_tfidf.nnz / total_reports if total_reports > 0 else 0

		# Consistency score: higher when fewer unique tokens relative to reports
		# This indicates more consistent vocabulary usage
		if total_reports > 1 and unique_tokens > 0:
			consistency_score = 1 - (unique_tokens / (total_reports * avg_tokens_per_report))
			consistency_score = max(0, consistency_score)	# Normalise to 0-1
		else:
			consistency_score = 0

		consistency_scores.append((category, unique_tokens, avg_tokens_per_report, consistency_score, total_reports))

		if total_reports >= 10:	# Only show categories with sufficient data
			print(f"{category[:25]:<25} | {unique_tokens:>8} | {avg_tokens_per_report:>12.1f} | {consistency_score:>12.3f}")

	# Sort by consistency score (lowest first = most inconsistent)
	consistency_scores.sort(key=lambda x: x[3])

	print(f"\nMOST INCONSISTENT VOCABULARIES (needs standardisation):")
	print("Category | Consistency Score | Sample Size")
	print("-"*50)
	for category, _, _, consistency, sample_size in consistency_scores[:10]:
		if sample_size >= 10:	# Only show meaningful sample sizes
			print(f"{category[:30]:<30} | {consistency:>13.3f} | {sample_size:>6}")

	return consistency_scores

def analyze_classification_quality(model_tfidf, df_filtered, X_tfidf, y, tfidf_vectorizer):
	"""Analyse quality and consistency of failure classifications"""
	print("="*50)
	print("CLASSIFICATION QUALITY AUDIT")
	print("="*50)

	# Get predictions for confusion matrix
	y_pred = cross_val_predict(model_tfidf, X_tfidf, y, cv=5)

	# Create a mapping of predictions to original indices
	prediction_indices = {}
	for idx, (true_label, pred_label) in enumerate(zip(y, y_pred)):
		original_idx = df_filtered.index[idx]
		if true_label not in prediction_indices:
			prediction_indices[true_label] = {'correct': [], 'incorrect': []}

		if true_label == pred_label:
			prediction_indices[true_label]['correct'].append(original_idx)
		else:
			prediction_indices[true_label]['incorrect'].append(original_idx)

	# 1. LOW CONFIDENCE CATEGORIES (poor performance)
	report = classification_report(y, y_pred, output_dict=True)

	low_confidence = []
	for category, metrics in report.items():
		if category not in ['accuracy', 'macro avg', 'weighted avg']:
			if metrics['f1-score'] < 0.7:	# Threshold for "problematic"
				low_confidence.append((category, metrics['f1-score'], metrics['support']))

	low_confidence.sort(key=lambda x: x[1])		# Sort by F1 score (worst first)

	print("\nLOW CONFIDENCE CATEGORIES (F1 < 0.7):")
	print("Category | F1 Score | Sample Size")
	print("-"*50)
	for category, f1, support in low_confidence:
		incorrect_samples = prediction_indices.get(category, {}).get('incorrect', [])
		print(f"{category[:30]:<30} | {f1:.3f} | {support} | Sample indices: {incorrect_samples[:5]}...")

	# 2. CONFUSED PAIRS (categories frequently misclassified as each other)
	labels = sorted(y.unique())
	conf_matrix = confusion_matrix(y, y_pred, labels=labels)

	confused_pairs = []
	for i in range(len(labels)):
		for j in range(len(labels)):
			if i != j and conf_matrix[i, j] > 5:	# At least 5 misclassifications
				confusion_rate = conf_matrix[i, j] / conf_matrix[i].sum()
				if confusion_rate > 0.1:	# More than 10% confusion rate
					confused_pairs.append((labels[i], labels[j], conf_matrix[i, j], confusion_rate))

	confused_pairs.sort(key=lambda x: x[3], reverse=True)	# Sort by confusion rate

	# Store examples of each confusion pair
	confusion_examples = {}
	for i, (true_cat, pred_cat) in enumerate(zip(y, y_pred)):
		if true_cat != pred_cat:
			pair = (true_cat, pred_cat)
			if pair not in confusion_examples:
				confusion_examples[pair] = []
			confusion_examples[pair].append(df_filtered.index[i])

	print(f"\nCONFUSED PAIRS (>10% misclassification rate):")
	print("True Category -> Predicted As | Count | Rate")
	print("-"*50)
	for true_cat, pred_cat, count, rate in confused_pairs[:10]:	# Top 10
		examples = confusion_examples.get((true_cat, pred_cat), [])[:3]
		print(f"{true_cat[:20]:<20} -> {pred_cat[:20]:<20} | {count:>3} | {rate:.1%} | Examples: {examples}")

	# 3. INCONSISTENT VOCABULARIES (same failure type, different token)
	feature_names = tfidf_vectorizer.get_feature_names_out()
	vocabulary_consistency = analyze_vocabulary_consistency(df_filtered, y, feature_names, X_tfidf)

	return low_confidence, confused_pairs, vocabulary_consistency, y_pred

def generate_business_recommendations(low_confidence, confused_pairs, vocabulary_consistency):
	"""Generate 80/20 recommendations for M&S teams"""

	print("\n" + "="*50)
	print("BUSINESS RECOMMENDATIONS (80/20 ANALYSIS)")
	print("="*50)

	print("\n🔴 HIGH PRIORITY (Immediate Action Required):")

	# Worst performing categories with sufficient sample size
	high_priority = [item for item in low_confidence if item[2] >= 20]	# At least 20 samples
	for category, f1, support in high_priority[:5]:
		print(f"  • {category}")
		print(f"	Issue: Poor classification accuracy ({f1:.1%})")
		print(f"	Impact: {support} incidents affected")
		print(f"	Action: Review documentation standards and provide targeted training")
		print()

	print("🟡 MEDIUM PRIORITY (Standardisation Needed):")

	# Most confused pairs
	for true_cat, pred_cat, count, rate in confused_pairs[:3]:
		print(f"  • '{true_cat}' often misclassified as '{pred_cat}' ({rate:.1%} of time)")
		print(f"	Action: Clarify distinction between these categories in training materials")
		print()

	# Most inconsistent vocabularies
	inconsistent = [item for item in vocabulary_consistency if item[4] >= 20]	# Sufficient sample
	for category, _, _, consistency, sample_size in inconsistent[:3]:
		print(f"  • {category}")
		print(f"	Issue: Highly inconsistent terminology (score: {consistency:.3f})")
		print(f"	Action: Develop standardised vocabulary guide")
		print()

	print("🟢 LOW PRIORITY (Monitor):")
	print("  • Categories with >80% accuracy and consistent vocabulary")
	print("  • Continue current documentation practices")

	return {
		'high_priority': high_priority,
		'confused_pairs': confused_pairs,
		'inconsistent_vocab': inconsistent
	}

def run_business_analysis(model_tfidf, df_filtered, X_tfidf, y, tfidf_vectorizer):
	"""Run complete business analysis"""
	low_confidence, confused_pairs, vocabulary_consistency, y_pred = analyze_classification_quality(
		model_tfidf, df_filtered, X_tfidf, y, tfidf_vectorizer
	)

	recommendations = generate_business_recommendations(low_confidence, confused_pairs, vocabulary_consistency)

	return low_confidence, confused_pairs, vocabulary_consistency, recommendations

if __name__ == "__main__":
	# Load and process
	df = load_raw_data()
	df_ml, tfidf_features, phrase_features, tfidf_vec, phrase_vec = feature_engineering(df)

	# Run initial clustering/classification
	model_tfidf, model_phrases, model_combined, meta_model, df_filtered, X_tfidf, X_phrases, y = initial_ml_clustering(
		df_ml, tfidf_features, phrase_features
	)

	print("Initial processing complete!")
	print("All models trained including meta-learner!")
	print("Starting Business Analysis!")
	run_business_analysis(model_tfidf, df_filtered, X_tfidf, y, tfidf_vec)

	# feature_names = tfidf_vec.get_feature_names_out()
	# importances = model_tfidf.feature_importances_
	# top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
	# print("Top 20 predictive tokens:")
	# for token, importance in top_features:
	# 	print(f"{token}: {importance:.4f}")
