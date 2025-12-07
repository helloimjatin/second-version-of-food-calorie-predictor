import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
import xgboost as xgb
import re
import random

# --- CONFIGURATION ---
DATA_FILE = 'final_merged_dataset_v2.csv'
MODEL_FILE = 'nutrition_model.pkl'
QUALITY_THRESHOLD = 60  # Lowered to get more data
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2  # Creates 2x more data

# --- DATA AUGMENTATION ---
def augment_data(df):
    """Create synthetic variations of existing data"""
    print("🔄 Augmenting dataset...")
    augmented = []
    
    # Cooking method synonyms
    cooking_synonyms = {
        'fried': ['deep fried', 'pan fried', 'stir fried'],
        'boiled': ['steamed', 'poached'],
        'grilled': ['roasted', 'baked'],
        'curry': ['gravy', 'masala'],
        'spicy': ['hot', 'chili'],
    }
    
    for idx, row in df.iterrows():
        # Original row
        augmented.append(row.to_dict())
        
        # Create variations
        for _ in range(AUGMENTATION_FACTOR - 1):
            new_row = row.to_dict()
            
            # Variation 1: Rephrase ingredients slightly
            ingredients = new_row['Ingredients']
            for orig, syns in cooking_synonyms.items():
                if orig in ingredients.lower():
                    replacement = random.choice(syns)
                    ingredients = re.sub(orig, replacement, ingredients, count=1, flags=re.IGNORECASE)
            new_row['Ingredients'] = ingredients
            
            # Variation 2: Add small realistic noise to nutrition (±5%)
            noise_factor = random.uniform(0.95, 1.05)
            new_row['Calories_100g'] *= noise_factor
            new_row['Protein_100g'] *= noise_factor
            new_row['Fat_100g'] *= noise_factor
            new_row['Carbs_100g'] *= noise_factor
            
            augmented.append(new_row)
    
    augmented_df = pd.DataFrame(augmented)
    print(f"   ✅ Dataset expanded from {len(df)} to {len(augmented_df)} rows")
    return augmented_df

# --- ENHANCED FEATURE ENGINEERING ---
class AdvancedIngredientFeatures(BaseEstimator, TransformerMixin):
    """Extract multiple features from ingredients text"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        features = []
        for text in X.astype(str):
            text_lower = text.lower()
            
            # Basic counts
            ingredient_count = text.count(',') + 1
            text_length = len(text)
            
            # Macro-nutrient keywords (expanded)
            protein_keywords = ['chicken', 'egg', 'meat', 'fish', 'paneer', 'dal', 
                              'lentil', 'tofu', 'protein', 'legume', 'bean', 'peas',
                              'milk', 'yogurt', 'curd', 'cottage cheese']
            protein_score = sum(1 for kw in protein_keywords if kw in text_lower)
            
            fat_keywords = ['oil', 'ghee', 'butter', 'cream', 'cheese', 'coconut', 
                          'nut', 'fried', 'cashew', 'almond', 'peanut', 'sesame',
                          'olive oil', 'vegetable oil', 'cooking oil']
            fat_score = sum(1 for kw in fat_keywords if kw in text_lower)
            
            carb_keywords = ['rice', 'bread', 'flour', 'sugar', 'potato', 'pasta', 
                           'wheat', 'maida', 'roti', 'naan', 'paratha', 'dosa',
                           'idli', 'upma', 'poha', 'corn', 'sweet']
            carb_score = sum(1 for kw in carb_keywords if kw in text_lower)
            
            # Cooking method (affects calories)
            is_fried = int(any(word in text_lower for word in ['fried', 'fry', 'deep fry']))
            is_baked = int(any(word in text_lower for word in ['baked', 'roasted', 'grilled']))
            is_raw = int(any(word in text_lower for word in ['raw', 'salad', 'fresh']))
            
            # Spice/seasoning level (often correlates with fat)
            spice_level = sum(1 for word in ['spices', 'masala', 'chili', 'pepper', 'curry'] 
                             if word in text_lower)
            
            features.append([
                ingredient_count, text_length, protein_score, fat_score, 
                carb_score, is_fried, is_baked, is_raw, spice_level
            ])
        
        return np.array(features)

def train():
    print("⏳ Loading dataset...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {DATA_FILE}.")
        return

    # 1. Filter with lower threshold
    df = df[df['Match_Score'] >= QUALITY_THRESHOLD].copy()
    print(f"   ✅ Loaded {len(df)} rows with Match_Score >= {QUALITY_THRESHOLD}")
    
    # 2. Apply data augmentation
    if USE_AUGMENTATION:
        df = augment_data(df)

    # 3. Feature engineering
    df['combined_text'] = df['Dish_Name'] + " " + df['Ingredients'].fillna('')
    df['raw_ingredients'] = df['Ingredients'].fillna('')
    df['dish_name_length'] = df['Dish_Name'].str.len()
    
    # NEW: Extract numeric features from Match_Score
    df['match_score_normalized'] = df['Match_Score'] / 100.0
    
    X = df[['combined_text', 'Diet_Type', 'raw_ingredients', 'dish_name_length', 'match_score_normalized']]
    y = df[['Calories_100g', 'Protein_100g', 'Fat_100g', 'Carbs_100g']]

    # 4. Advanced preprocessing
    text_pipe = TfidfVectorizer(
        stop_words='english',
        max_features=1500,
        ngram_range=(1, 3),  # Up to trigrams
        min_df=2,
        max_df=0.8,  # Remove very common words
        sublinear_tf=True
    )
    
    cat_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    ingredient_features_pipe = Pipeline([
        ('features', AdvancedIngredientFeatures()),
        ('scaler', RobustScaler())  # Better for outliers
    ])
    
    numeric_pipe = RobustScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipe, 'combined_text'),
            ('cat', cat_pipe, ['Diet_Type']),
            ('ingredient_features', ingredient_features_pipe, 'raw_ingredients'),
            ('numeric', numeric_pipe, ['dish_name_length', 'match_score_normalized'])
        ])

    # 5. Train with stratified split
    print("🔪 Splitting data (stratified by Diet_Type)...")
    
    # Create bins for stratification
    df['calorie_bin'] = pd.qcut(df['Calories_100g'], q=4, labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=df['calorie_bin']
    )
    
    # Preprocess
    print("🔧 Preprocessing features...")
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Train ensemble of models
    models = {}
    target_names = ['Calories_100g', 'Protein_100g', 'Fat_100g', 'Carbs_100g']
    predictions = {}
    
    print("\n🚀 Training optimized models...")
    
    for i, target in enumerate(target_names):
        print(f"\n   Training {target}...")
        
        # Use different models for different targets
        if target == 'Fat_100g':
            # Fat is hardest - use ensemble with log transform
            base_model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1
            )
            # Log transform for fat (handles wide range better)
            y_train_log = np.log1p(y_train.iloc[:, i])
            base_model.fit(X_train_processed, y_train_log)
            y_pred_log = base_model.predict(X_test_processed)
            y_pred = np.expm1(y_pred_log)  # Transform back
            
        elif target == 'Calories_100g':
            # Calories - use Ridge with higher regularization
            model = Ridge(alpha=50.0)
            model.fit(X_train_processed, y_train.iloc[:, i])
            y_pred = model.predict(X_test_processed)
            
        else:
            # For Protein and Carbs, use XGBoost
            model = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_processed, y_train.iloc[:, i])
            y_pred = model.predict(X_test_processed)
        
        # Post-processing: clip predictions to realistic ranges
        if target == 'Protein_100g':
            y_pred = np.clip(y_pred, 0, 30)
        elif target == 'Fat_100g':
            y_pred = np.clip(y_pred, 0, 100)
        elif target == 'Carbs_100g':
            y_pred = np.clip(y_pred, 0, 100)
        elif target == 'Calories_100g':
            y_pred = np.clip(y_pred, 0, 900)
        
        predictions[target] = y_pred
        
        # Evaluate
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred)
        r2 = r2_score(y_test.iloc[:, i], y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test.iloc[:, i] - y_pred) / (y_test.iloc[:, i] + 1))) * 100
        
        print(f"      MAE: ±{mae:.2f} | R²: {r2:.3f} | MAPE: {mape:.1f}%")
        
        # Store the model (for Fat, need to store the special handling)
        if target == 'Fat_100g':
            models[target] = base_model
            models[target + '_log_transform'] = True
        else:
            models[target] = model
    
    # 7. Calculate calories from macros (more accurate!)
    print("\n🧮 Recalculating calories from macronutrients...")
    calories_calculated = (predictions['Protein_100g'] * 4 + 
                          predictions['Fat_100g'] * 9 + 
                          predictions['Carbs_100g'] * 4)
    predictions['Calories_100g'] = calories_calculated
    
    # 8. Overall evaluation
    y_pred_all = np.column_stack([predictions[t] for t in target_names])
    overall_r2 = r2_score(y_test, y_pred_all)
    
    print("\n" + "="*60)
    print("📊 FINAL MODEL PERFORMANCE")
    print("="*60)
    print("(Calories calculated from: Protein×4 + Fat×9 + Carbs×4)")
    print("-"*60)
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_test.iloc[:, i], predictions[target])
        r2 = r2_score(y_test.iloc[:, i], predictions[target])
        mape = np.mean(np.abs((y_test.iloc[:, i] - predictions[target]) / (y_test.iloc[:, i] + 1))) * 100
        print(f"{target:15s} | MAE: ±{mae:5.2f} | R²: {r2:.3f} | MAPE: {mape:4.1f}%")
    print(f"\nOverall R² Score: {overall_r2:.3f}")
    print("="*60)
    
    # Interpretation
    print("\n💡 PERFORMANCE INTERPRETATION:")
    if overall_r2 > 0.65:
        print("   ✅ EXCELLENT - Model is production-ready!")
    elif overall_r2 > 0.5:
        print("   ✅ GOOD - Model is usable for real applications")
    elif overall_r2 > 0.35:
        print("   ⚠️  FAIR - Model has learned patterns but has limitations")
    else:
        print("   ❌ POOR - Model needs more data or different approach")
    
    print(f"\n   With {len(df)} training samples:")
    print(f"   • Calorie predictions: ±{mean_absolute_error(y_test.iloc[:, 0], predictions['Calories_100g']):.0f} kcal")
    print(f"   • Protein predictions: ±{mean_absolute_error(y_test.iloc[:, 1], predictions['Protein_100g']):.1f}g")

    # 9. Save model using pickle
    print(f"\n💾 Saving model to {MODEL_FILE}...")
    package = {
        'preprocessor': preprocessor,
        'models': models,
        'target_names': target_names,
        'database': df,
        'use_calorie_formula': True  # Flag to calculate calories from macros
    }
    
    # Save using pickle with highest protocol for better compatibility
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ Training Complete!")
    print("\n📝 Note: Calories are calculated from macros for better accuracy")

if __name__ == "__main__":
    train()