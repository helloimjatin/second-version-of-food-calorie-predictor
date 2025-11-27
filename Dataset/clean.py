import pandas as pd
import os

# --- 1. CONFIGURATION ---
# Make sure these filenames match YOUR files exactly
FILE_NUTRITION = 'INDB.xlsx'
FILE_RECIPES = 'IndianFoodDatasetCSV.csv'

OUTPUT_NUTRITION = 'cleaned_nutrition_data.csv'
OUTPUT_RECIPES = 'cleaned_recipe_text.csv'

def robust_load(filepath):
    """
    Helper function to load files even if they have weird formats or errors.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: '{filepath}'")
        return None

    # Try different ways to open the file (Excel vs CSV, UTF-8 vs Latin-1)
    loaders = [
        (pd.read_csv, {'encoding': 'utf-8'}),
        (pd.read_csv, {'encoding': 'latin-1'}),
        (pd.read_csv, {'sep': None, 'engine': 'python', 'encoding': 'latin-1'}),
        (pd.read_excel, {})
    ]

    for loader, args in loaders:
        try:
            return loader(filepath, **args)
        except:
            continue
            
    print(f"❌ Critical: Could not read '{filepath}'. Check if it's corrupted.")
    return None

def clean_nutrition():
    print(f"\n🔄 Cleaning Nutrition Data ({FILE_NUTRITION})...")
    df = robust_load(FILE_NUTRITION)
    if df is None: return

    # --- REQUIREMENT 1: REMOVE UNWANTED COLUMNS ---
    # We define ONLY the columns we want to keep.
    # Everything else (micronutrients, vitamins) will be dropped automatically.
    target_cols = {
        'food_name': 'Dish_Name',               # Will become Dish_Name
        'energy_kcal': 'Calories_100g',         # Will become Calories_100g
        'protein_g': 'Protein_100g',
        'fat_g': 'Fat_100g',
        'carb_g': 'Carbs_100g',
        'servings_unit': 'Portion_Unit',
        'unit_serving_energy_kcal': 'Calories_Portion'
    }

    # Filter: Keep only columns that exist in the file
    available_cols = [c for c in target_cols.keys() if c in df.columns]
    df_clean = df[available_cols].copy()

    # --- REQUIREMENT 2: STANDARDIZE COLUMN NAMES ---
    # Rename columns so they match the other dataset
    df_clean.rename(columns=target_cols, inplace=True)

    # Clean Text: Lowercase and strip spaces for perfect merging later
    if 'Dish_Name' in df_clean.columns:
        df_clean['Dish_Name'] = df_clean['Dish_Name'].astype(str).str.lower().str.strip()

    # Fill empty numbers with 0
    numeric_cols = ['Calories_100g', 'Protein_100g', 'Fat_100g', 'Carbs_100g']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # Save
    df_clean.to_csv(OUTPUT_NUTRITION, index=False)
    print(f"✅ Success! Saved {OUTPUT_NUTRITION} (Columns: {list(df_clean.columns)})")


def clean_recipes():
    print(f"\n🔄 Cleaning Recipe Data ({FILE_RECIPES})...")
    df = robust_load(FILE_RECIPES)
    if df is None: return

    # --- REQUIREMENT 1: REMOVE UNWANTED COLUMNS ---
    # We only want Name, Ingredients, and Diet.
    # We drop Instructions, PrepTime, URL, etc.
    cols_to_keep = ['RecipeName', 'Diet']
    
    # Handle variations in Ingredient column names
    if 'Ingredients' in df.columns:
        cols_to_keep.append('Ingredients')
        ing_col = 'Ingredients'
    elif 'TranslatedIngredients' in df.columns:
        cols_to_keep.append('TranslatedIngredients')
        ing_col = 'TranslatedIngredients'
    else:
        print("⚠️ Warning: Could not find Ingredients column.")
        return

    df_clean = df[cols_to_keep].copy()

    # --- REQUIREMENT 2: STANDARDIZE COLUMN NAMES ---
    # Rename 'RecipeName' -> 'Dish_Name' to match the Nutrition file
    rename_map = {
        'RecipeName': 'Dish_Name',
        ing_col: 'Ingredients_Text',
        'Diet': 'Diet_Type'
    }
    df_clean.rename(columns=rename_map, inplace=True)

    # Clean Text
    df_clean['Dish_Name'] = df_clean['Dish_Name'].astype(str).str.lower().str.strip()

    # Save
    df_clean.to_csv(OUTPUT_RECIPES, index=False)
    print(f"✅ Success! Saved {OUTPUT_RECIPES} (Columns: {list(df_clean.columns)})")

if __name__ == "__main__":
    clean_nutrition()
    clean_recipes()
    print("\n🎉 Datasets cleaned! Now both have 'Dish_Name' and only necessary columns.")