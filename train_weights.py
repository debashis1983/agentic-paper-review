#!/usr/bin/env python3
"""
Train Regression Weights for Paper Review Scoring
==================================================

This script trains a linear regression model to predict review ratings
from dimensional scores (soundness, presentation, contribution).


Confidence handling:
1. As predictor: Include confidence as a feature
2. As weight: Use confidence to weight samples in regression
3. Exclude: Don't use confidence at all

"""

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load CSV and validate required columns."""
    df = pd.read_csv(filepath)
    
    required_cols = ['paper_id', 'rating', 'soundness', 'presentation', 'contribution']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with missing values in key columns
    original_len = len(df)
    df = df.dropna(subset=['rating', 'soundness', 'presentation', 'contribution'])
    if len(df) < original_len:
        print(f"  Dropped {original_len - len(df)} rows with missing values")
    
    return df


def prepare_individual_data(df: pd.DataFrame, use_confidence: str) -> tuple:
    """Prepare data at individual review level."""
    features = ['soundness', 'presentation', 'contribution']
    
    if use_confidence == 'predictor' and 'confidence' in df.columns:
        features.append('confidence')
        df = df.dropna(subset=['confidence'])
    
    X = df[features].values
    y = df['rating'].values
    
    # Sample weights (for weighted regression)
    if use_confidence == 'weight' and 'confidence' in df.columns:
        weights = df['confidence'].fillna(df['confidence'].median()).values
    else:
        weights = None
    
    return X, y, weights, features


def prepare_averaged_data(df: pd.DataFrame, use_confidence: str) -> tuple:
    """Prepare data averaged per paper."""
    agg_cols = {
        'rating': 'mean',
        'soundness': 'mean',
        'presentation': 'mean',
        'contribution': 'mean',
    }
    
    if 'confidence' in df.columns:
        agg_cols['confidence'] = 'mean'
    
    # Also count reviews per paper
    df_avg = df.groupby('paper_id').agg(agg_cols).reset_index()
    df_avg['num_reviews'] = df.groupby('paper_id').size().values
    
    print(f"  Averaged to {len(df_avg)} papers (from {len(df)} reviews)")
    print(f"  Reviews per paper: {df_avg['num_reviews'].mean():.1f} avg, {df_avg['num_reviews'].min()}-{df_avg['num_reviews'].max()} range")
    
    features = ['soundness', 'presentation', 'contribution']
    
    if use_confidence == 'predictor' and 'confidence' in df_avg.columns:
        features.append('confidence')
    
    X = df_avg[features].values
    y = df_avg['rating'].values
    
    # Weight by number of reviews (more reviews = more reliable average)
    if use_confidence == 'weight':
        weights = df_avg['num_reviews'].values
    else:
        weights = None
    
    return X, y, weights, features


def train_model(X: np.ndarray, y: np.ndarray, weights: np.ndarray = None, 
                regularization: float = 0.0) -> LinearRegression:
    """Train linear regression model."""
    if regularization > 0:
        model = Ridge(alpha=regularization)
    else:
        model = LinearRegression()
    
    model.fit(X, y, sample_weight=weights)
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray, features: list) -> dict:
    """Evaluate model performance."""
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(y, y_pred)
    spearman_r, spearman_p = spearmanr(y, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
    }


def print_results(model, features: list, metrics: dict, cv_scores: np.ndarray = None):
    """Print training results."""
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    # Coefficients
    print("\nLearned Weights (Coefficients):")
    print("-" * 40)
    
    total_weight = sum(abs(c) for c in model.coef_)
    for feat, coef in zip(features, model.coef_):
        pct = abs(coef) / total_weight * 100
        print(f"  {feat:20s}: {coef:+.4f} ({pct:.1f}%)")
    print(f"  {'intercept':20s}: {model.intercept_:+.4f}")
    
    # Normalized weights (sum to 1)
    print("\nNormalized Weights (for prediction):")
    print("-" * 40)
    coef_sum = sum(model.coef_)
    for feat, coef in zip(features, model.coef_):
        norm_weight = coef / coef_sum if coef_sum != 0 else 0
        print(f"  {feat:20s}: {norm_weight:.4f}")
    
    # Metrics
    print("\nModel Performance:")
    print("-" * 40)
    print(f"  RMSE:              {metrics['rmse']:.4f}")
    print(f"  R² Score:          {metrics['r2']:.4f}")
    print(f"  Pearson r:         {metrics['pearson_r']:.4f}")
    print(f"  Spearman ρ:        {metrics['spearman_r']:.4f}")
    
    if cv_scores is not None:
        print(f"\n  Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Compare to paper
    print("\nComparison to Andrew Ng's work and dsicussions:")
    print("-" * 40)
    print(f"  Paper AI-Human Spearman: 0.42")
    print(f"  Our Spearman:            {metrics['spearman_r']:.2f}")
    if metrics['spearman_r'] >= 0.40:
        print("  ✅ Comparable to paper results!")
    else:
        print("  ⚠️  Lower than paper (may need more data or features)")


def save_weights(model, features: list, metrics: dict, output_path: str):
    """Save learned weights to JSON."""
    weights = {feat: float(coef) for feat, coef in zip(features, model.coef_)}
    
    # Also save normalized weights
    coef_sum = sum(model.coef_)
    normalized = {feat: float(coef/coef_sum) for feat, coef in zip(features, model.coef_)}
    
    output = {
        'weights': weights,
        'normalized_weights': normalized,
        'intercept': float(model.intercept_),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'features': features,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved weights to: {output_path}")


def generate_agent_code(model, features: list):
    """Generate code snippet for agent.py."""
    print("\n" + "="*60)
    print("CODE SNIPPET FOR agent.py")
    print("="*60)
    print("""
# Learned weights from training on NeurIPS/ICLR data
DIMENSION_WEIGHTS = {""")
    
    coef_sum = sum(model.coef_)
    for feat, coef in zip(features, model.coef_):
        norm_weight = coef / coef_sum if coef_sum != 0 else 0
        # Map to 7-dimension names
        if feat == 'soundness':
            print(f'    "Experimental_Soundness": {norm_weight:.4f},')
            print(f'    "Claims_Support": {norm_weight:.4f},  # Same as soundness')
        elif feat == 'presentation':
            print(f'    "Clarity": {norm_weight:.4f},')
        elif feat == 'contribution':
            print(f'    "Importance": {norm_weight:.4f},')
            print(f'    "Community_Value": {norm_weight:.4f},  # Same as contribution')
        elif feat == 'confidence':
            print(f'    # Confidence: {norm_weight:.4f} (used as predictor)')
    
    print("""}

def calculate_final_score(dimension_scores: dict) -> float:
    \"\"\"Calculate weighted final score from dimensions.\"\"\"
    score = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        if dim in dimension_scores:
            score += weight * dimension_scores[dim]
    return score
""")


def main():
    parser = argparse.ArgumentParser(description="Train regression weights for paper review scoring")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with reviews")
    parser.add_argument("--output", "-o", default="learned_weights.json", help="Output JSON file")
    parser.add_argument("--mode", "-m", choices=["individual", "averaged"], default="individual",
                        help="Training mode: individual reviews or averaged per paper")
    parser.add_argument("--confidence", "-c", choices=["predictor", "weight", "exclude"], 
                        default="exclude", help="How to use confidence score")
    parser.add_argument("--regularization", "-r", type=float, default=0.0,
                        help="Ridge regularization strength (0 = no regularization)")
    parser.add_argument("--test-split", "-t", type=float, default=0.2,
                        help="Test set proportion (0-1)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Paper Review Score Regression Training")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Input:        {args.input}")
    print(f"  Mode:         {args.mode}")
    print(f"  Confidence:   {args.confidence}")
    print(f"  Regularization: {args.regularization}")
    
    # Load data
    print(f"\nLoading data...")
    df = load_and_validate_data(args.input)
    print(f"  Loaded {len(df)} reviews for {df['paper_id'].nunique()} papers")
    
    # Data statistics
    print(f"\nData Statistics:")
    print(f"  Rating:       {df['rating'].mean():.2f} ± {df['rating'].std():.2f} (range: {df['rating'].min()}-{df['rating'].max()})")
    print(f"  Soundness:    {df['soundness'].mean():.2f} ± {df['soundness'].std():.2f}")
    print(f"  Presentation: {df['presentation'].mean():.2f} ± {df['presentation'].std():.2f}")
    print(f"  Contribution: {df['contribution'].mean():.2f} ± {df['contribution'].std():.2f}")
    if 'confidence' in df.columns:
        print(f"  Confidence:   {df['confidence'].mean():.2f} ± {df['confidence'].std():.2f}")
    
    # Prepare data based on mode
    print(f"\nPreparing data ({args.mode} mode)...")
    if args.mode == "individual":
        X, y, weights, features = prepare_individual_data(df, args.confidence)
    else:
        X, y, weights, features = prepare_averaged_data(df, args.confidence)
    
    print(f"  Features: {features}")
    print(f"  Samples:  {len(y)}")
    print(f"  Weights:  {'Yes' if weights is not None else 'No'}")
    
    # Split data
    if args.test_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_split, random_state=42
        )
        if weights is not None:
            w_train, w_test = train_test_split(weights, test_size=args.test_split, random_state=42)
        else:
            w_train, w_test = None, None
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        w_train, w_test = weights, weights
    
    # Train model
    print(f"\nTraining model...")
    model = train_model(X_train, y_train, w_train, args.regularization)
    
    # Evaluate
    metrics_train = evaluate_model(model, X_train, y_train, features)
    metrics_test = evaluate_model(model, X_test, y_test, features)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Print results
    print_results(model, features, metrics_test, cv_scores)
    
    # Save weights
    save_weights(model, features, metrics_test, args.output)
    
    # Generate code snippet
    generate_agent_code(model, features)
    
    # Correlation analysis
    print("\n" + "="*60)
    print("FEATURE CORRELATIONS WITH RATING")
    print("="*60)
    for i, feat in enumerate(features):
        corr, _ = pearsonr(X[:, i], y)
        print(f"  {feat:20s}: r = {corr:.3f}")


if __name__ == "__main__":
    main()