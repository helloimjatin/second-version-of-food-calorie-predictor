import pandas as pd
from thefuzz import process, fuzz

# --- CONFIGURATION ---
FILE_NUT = 'cleaned_nutrition_data.csv'
FILE_REC = 'cleaned_recipe_text.csv'
OUTPUT_FILE = 'final_merged_dataset_v2.csv'

# Lower threshold because token_set_ratio is smarter but sometimes produces lower raw scores
MATCH_THRESHOLD = 65

def advanced_clean(name):
    """
    Aggressive cleaning to strip descriptions.
    "Masala Karela - Diabetic Friendly" -> "masala karela"
    "Potato/Aloo Paratha" -> "potato" (takes the first one)
    """
    name = str(name).lower()
    
    # split on common separators and take the first part
    separators = ['(', '-', '/', ',', ':']
    for sep in separators:
        if sep in name:
            name = name.split(sep)[0]
    
    # remove noise words
    noise_words = ['recipe', 'authentic', 'indian', 'style', 'how to make', 'spicy', 'curry']
    for word in noise_words:
        name = name.replace(word, '')
        
    return name.strip()

def merge_datasets():
    print("⏳ Loading datasets...")
    df_nut = pd.read_csv(FILE_NUT)
    df_rec = pd.read_csv(FILE_REC)
    
    # 1. Apply Advanced Cleaning
    print("🧹 applying advanced name cleaning...")
    df_nut['match_name'] = df_nut['Dish_Name'].apply(advanced_clean)
    df_rec['match_name'] = df_rec['Dish_Name'].apply(advanced_clean)
    
    # Remove rows where name became empty after cleaning
    df_rec = df_rec[df_rec['match_name'].str.len() > 2]
    
    recipe_choices = df_rec['match_name'].tolist()
    
    print("🔗 Starting Smart Merge (using token_set_ratio)...")
    
    matches = []
    
    for index, row in df_nut.iterrows():
        nut_name = row['match_name']
        
        # KEY UPGRADE: Use token_set_ratio instead of sort_ratio
        # This matches "Aloo Paratha" with "Punjabi Aloo Paratha" perfectly (Score 100)
        best_match = process.extractOne(nut_name, recipe_choices, scorer=fuzz.token_set_ratio)
        
        if best_match:
            match_name, score = best_match
            
            if score >= MATCH_THRESHOLD:
                # Find the original row in recipe DB
                matched_row = df_rec[df_rec['match_name'] == match_name].iloc[0]
                
                matches.append({
                    'Dish_Name': row['Dish_Name'], # Name from Nutrition DB
                    'Calories_100g': row['Calories_100g'],
                    'Protein_100g': row['Protein_100g'],
                    'Fat_100g': row['Fat_100g'],
                    'Carbs_100g': row['Carbs_100g'],
                    'Portion_Unit': row['Portion_Unit'],
                    'Calories_Portion': row['Calories_Portion'],
                    
                    # Merged Data
                    'Matched_Recipe_Name': matched_row['Dish_Name'],
                    'Ingredients': matched_row['Ingredients_Text'],
                    'Diet_Type': matched_row['Diet_Type'],
                    'Match_Score': score
                })
    
    final_df = pd.DataFrame(matches)
    
    # Remove duplicates (sometimes multiple nutrition items match the same recipe)
    final_df.drop_duplicates(subset=['Dish_Name'], inplace=True)
    
    print("-" * 30)
    print(f"✅ Merge Complete!")
    print(f"   Original Nutrition Items: {len(df_nut)}")
    print(f"   Successfully Matched: {len(final_df)}")
    if len(df_nut) > 0:
        print(f"   Match Rate: {round((len(final_df)/len(df_nut))*100, 1)}%")
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_datasets() 