import streamlit as st
import pandas as pd
import numpy as np
import pickle
import difflib
import xgboost as xgb  # ✅ REQUIRED: Added this because your model uses XGBoost
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. MUST PASTE CUSTOM CLASSES HERE TO LOAD MODEL ---
# (Pickle needs this class definition to exist in the namespace)
class AdvancedIngredientFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        features = []
        for text in X.astype(str):
            text_lower = text.lower()
            ingredient_count = text.count(',') + 1
            text_length = len(text)
            protein_keywords = ['chicken', 'egg', 'meat', 'fish', 'paneer', 'dal', 'lentil', 'tofu', 'protein', 'legume', 'bean', 'peas', 'milk', 'yogurt', 'curd', 'cottage cheese']
            protein_score = sum(1 for kw in protein_keywords if kw in text_lower)
            fat_keywords = ['oil', 'ghee', 'butter', 'cream', 'cheese', 'coconut', 'nut', 'fried', 'cashew', 'almond', 'peanut', 'sesame', 'olive oil', 'vegetable oil', 'cooking oil']
            fat_score = sum(1 for kw in fat_keywords if kw in text_lower)
            carb_keywords = ['rice', 'bread', 'flour', 'sugar', 'potato', 'pasta', 'wheat', 'maida', 'roti', 'naan', 'paratha', 'dosa', 'idli', 'upma', 'poha', 'corn', 'sweet']
            carb_score = sum(1 for kw in carb_keywords if kw in text_lower)
            is_fried = int(any(word in text_lower for word in ['fried', 'fry', 'deep fry']))
            is_baked = int(any(word in text_lower for word in ['baked', 'roasted', 'grilled']))
            is_raw = int(any(word in text_lower for word in ['raw', 'salad', 'fresh']))
            spice_level = sum(1 for word in ['spices', 'masala', 'chili', 'pepper', 'curry'] if word in text_lower)
            features.append([ingredient_count, text_length, protein_score, fat_score, carb_score, is_fried, is_baked, is_raw, spice_level])
        return np.array(features)

# --- 2. CONFIGURATION & LOADING ---
MODEL_FILE = 'nutrition_model.pkl'

@st.cache_resource
def load_model():
    try:
        # Load using pickle instead of joblib
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Could not find {MODEL_FILE}. Please run train_model.py first!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

data = load_model()

# --- 3. HELPER FUNCTIONS ---
def get_prediction(dish_name, diet_type):
    if not data: return None
    
    # Prepare input dataframe
    input_df = pd.DataFrame({
        'combined_text': [dish_name], # We don't have ingredients, so we just use name
        'Diet_Type': [diet_type],
        'raw_ingredients': [dish_name], # Fallback to using name as ingredient text
        'dish_name_length': [len(dish_name)],
        'match_score_normalized': [0.5] # Neutral score for fresh prediction
    })
    
    # Transform features
    try:
        X_processed = data['preprocessor'].transform(input_df)
    except Exception as e:
        st.error(f"Error transforming input: {e}")
        return None
    
    results = {}
    
    # Predict each target
    for target in data['target_names']:
        model = data['models'].get(target)
        
        # Handle special 'Fat' log transformation if it exists
        if target == 'Fat_100g' and 'Fat_100g_log_transform' in data['models']:
             # If your training script saved the base model, we might need manual predict
             # But usually pipeline handles it. Let's assume standard predict for now.
             pred = model.predict(X_processed)[0]
             # If log transformed, we need to expm1 it back? 
             # The training script logic for 'Fat' used a log transform manually before fitting.
             # If the model saved is just the XGBRegressor, it will output LOG values.
             # We need to reverse it.
             pred = np.expm1(pred)
        else:
             pred = model.predict(X_processed)[0]
        
        # Clip negatives
        results[target] = max(0.1, pred)

    # Calculate Calories from macros if flag is set
    if data.get('use_calorie_formula'):
        results['Calories_100g'] = (results['Protein_100g'] * 4) + \
                                   (results['Fat_100g'] * 9) + \
                                   (results['Carbs_100g'] * 4)
        
    return results

def find_database_match(user_input, db):
    # Search the database saved in the pickle
    matches = difflib.get_close_matches(user_input.lower(), db['Dish_Name'].astype(str).str.lower(), n=1, cutoff=0.85)
    if matches:
        match_name = matches[0]
        row = db[db['Dish_Name'].str.lower() == match_name].iloc[0]
        return row
    return None

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI Nutritionist", page_icon="🥗", layout="centered")

st.title("🥗 Indian Food Calorie AI")
st.markdown("Predict calories for **Indian Dishes** using Machine Learning.")

# Inputs
col1, col2 = st.columns([2, 1])
with col1:
    dish_name = st.text_input("Dish Name", placeholder="e.g. Butter Chicken, Masala Dosa")
with col2:
    quantity_g = st.number_input("Quantity (grams)", min_value=10, value=200, step=10)

diet_type = st.radio("Diet Type", ["Vegetarian", "Non Vegeterian", "Eggetarian"], horizontal=True)

if st.button("Analyze Food 🔍", type="primary"):
    if not dish_name:
        st.warning("Please enter a dish name.")
    else:
        st.divider()
        
        # Initialize vals to None
        vals = None
        source = None
        db_match = None

        # 1. Try Database Lookup First (High Accuracy)
        db_match = find_database_match(dish_name, data['database'])
        
        if db_match is not None:
            st.success(f"✅ Found exact match: **{db_match['Dish_Name'].title()}**")
            vals = {
                'Calories_100g': db_match['Calories_100g'],
                'Protein_100g': db_match['Protein_100g'],
                'Fat_100g': db_match['Fat_100g'],
                'Carbs_100g': db_match['Carbs_100g']
            }
            source = "Database (Verified)"
        else:
            st.info(f"🤖 No exact match. Predicting using AI Model...")
            vals = get_prediction(dish_name, diet_type)
            source = "AI Prediction (Estimated)"

        if vals:
            # 2. Calculate Totals based on Quantity
            # FIX: Ensure factor is defined before use
            factor = quantity_g / 100
            
            total_cal = vals['Calories_100g'] * factor
            total_pro = vals['Protein_100g'] * factor
            total_fat = vals['Fat_100g'] * factor
            total_carb = vals['Carbs_100g'] * factor

            # 3. Display Results
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.metric(label="🔥 Total Calories", value=f"{int(total_cal)} kcal")
                st.caption(f"Source: {source}")
                
            with col_b:
                st.write("#### Macro Breakdown")
                # Avoid division by zero
                total_macro = total_pro + total_fat + total_carb
                if total_macro > 0:
                    st.write(f"**Protein:** {total_pro:.1f}g")
                    st.progress(total_pro / total_macro)
                    
                    st.write(f"**Fat:** {total_fat:.1f}g")
                    st.progress(total_fat / total_macro)
                    
                    st.write(f"**Carbs:** {total_carb:.1f}g")
                    st.progress(total_carb / total_macro)
                else:
                    st.write("No macronutrients detected.")

            # 4. Ingredients Hint
            if db_match is not None and pd.notna(db_match.get('Ingredients')):
                 with st.expander("See Ingredients"):
                     st.write(db_match['Ingredients'])
