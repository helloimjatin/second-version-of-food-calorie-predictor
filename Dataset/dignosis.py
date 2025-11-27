import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = 'final_merged_dataset_v2.csv'
QUALITY_THRESHOLD = 70

def diagnose_data():
    print("🔍 NUTRITION DATA DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    df_filtered = df[df['Match_Score'] >= QUALITY_THRESHOLD].copy()
    
    print(f"\n📊 DATASET SIZE")
    print(f"   Total rows: {len(df)}")
    print(f"   High-quality rows (Match_Score >= {QUALITY_THRESHOLD}): {len(df_filtered)}")
    print(f"   ⚠️  Dataset is {'SMALL' if len(df_filtered) < 1000 else 'adequate'}")
    
    # Check for missing values
    print(f"\n❓ MISSING VALUES")
    target_cols = ['Calories_100g', 'Protein_100g', 'Fat_100g', 'Carbs_100g']
    feature_cols = ['Dish_Name', 'Ingredients', 'Diet_Type']
    
    for col in target_cols + feature_cols:
        missing = df_filtered[col].isna().sum()
        pct = (missing / len(df_filtered)) * 100
        status = "✅" if missing == 0 else "⚠️"
        print(f"   {status} {col:20s}: {missing:4d} missing ({pct:.1f}%)")
    
    # Check target distribution
    print(f"\n📈 TARGET DISTRIBUTION (should be realistic)")
    for col in target_cols:
        data = df_filtered[col].dropna()
        print(f"\n   {col}:")
        print(f"      Mean:   {data.mean():.1f}")
        print(f"      Median: {data.median():.1f}")
        print(f"      Std:    {data.std():.1f}")
        print(f"      Min:    {data.min():.1f}")
        print(f"      Max:    {data.max():.1f}")
        
        # Check for unrealistic values
        if col == 'Calories_100g':
            if data.min() < 0 or data.max() > 900:
                print(f"      ⚠️  UNREALISTIC values detected!")
        elif col in ['Protein_100g', 'Fat_100g', 'Carbs_100g']:
            if data.min() < 0 or data.max() > 100:
                print(f"      ⚠️  Values over 100g/100g detected (impossible!)")
    
    # Check variance
    print(f"\n📊 TARGET VARIANCE (low variance = hard to predict)")
    for col in target_cols:
        data = df_filtered[col].dropna()
        cv = (data.std() / data.mean()) * 100  # Coefficient of variation
        status = "✅" if cv > 30 else "⚠️"
        print(f"   {status} {col:20s}: CV = {cv:.1f}%")
    
    # Check feature quality
    print(f"\n🔤 FEATURE QUALITY")
    
    # Dish names
    avg_dish_len = df_filtered['Dish_Name'].str.len().mean()
    print(f"   Dish Name avg length: {avg_dish_len:.0f} chars")
    
    # Ingredients
    avg_ing_len = df_filtered['Ingredients'].fillna('').str.len().mean()
    avg_ing_count = df_filtered['Ingredients'].fillna('').apply(lambda x: x.count(',') + 1).mean()
    print(f"   Ingredients avg length: {avg_ing_len:.0f} chars")
    print(f"   Ingredients avg count: {avg_ing_count:.1f}")
    
    if avg_ing_len < 50:
        print(f"   ⚠️  Ingredients are too SHORT - not enough info!")
    
    # Diet type distribution
    print(f"\n🥗 DIET TYPE DISTRIBUTION")
    diet_counts = df_filtered['Diet_Type'].value_counts()
    for diet, count in diet_counts.items():
        pct = (count / len(df_filtered)) * 100
        print(f"   {diet:15s}: {count:4d} ({pct:.1f}%)")
    
    # Check for duplicates
    print(f"\n🔁 DUPLICATE CHECK")
    dup_dishes = df_filtered['Dish_Name'].duplicated().sum()
    print(f"   Duplicate dish names: {dup_dishes}")
    
    # Correlation check
    print(f"\n🔗 TARGET CORRELATION (should be moderate)")
    corr_matrix = df_filtered[target_cols].corr()
    print("\n" + str(corr_matrix.round(2)))
    
    # Check Match_Score distribution
    print(f"\n⭐ MATCH SCORE DISTRIBUTION")
    score_bins = [70, 80, 90, 100]
    for i in range(len(score_bins) - 1):
        count = ((df_filtered['Match_Score'] >= score_bins[i]) & 
                 (df_filtered['Match_Score'] < score_bins[i+1])).sum()
        print(f"   {score_bins[i]}-{score_bins[i+1]}: {count} rows")
    count = (df_filtered['Match_Score'] >= 90).sum()
    print(f"   90-100: {count} rows")
    
    # Identify potential issues
    print(f"\n🚨 POTENTIAL ISSUES IDENTIFIED:")
    issues = []
    
    if len(df_filtered) < 1000:
        issues.append("⚠️  Dataset too small (< 1000 rows)")
    
    if avg_ing_len < 50:
        issues.append("⚠️  Ingredients lack detail")
    
    for col in target_cols:
        missing = df_filtered[col].isna().sum()
        if missing > 0:
            issues.append(f"⚠️  Missing values in {col}")
    
    for col in target_cols:
        data = df_filtered[col].dropna()
        cv = (data.std() / data.mean()) * 100
        if cv < 30:
            issues.append(f"⚠️  Low variance in {col} (CV={cv:.1f}%)")
    
    if len(issues) == 0:
        print("   ✅ No major issues detected!")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Collect more data (aim for 2000+ samples)")
    print(f"   2. Ensure ingredients are detailed (50+ chars)")
    print(f"   3. Verify nutrition values are realistic")
    print(f"   4. Remove/fix any impossible values (e.g., >100g/100g)")
    print(f"   5. Consider data augmentation if stuck with small dataset")
    
    print("\n" + "=" * 60)
    
    # Save sample of problematic rows
    print("\n📝 Saving sample data for inspection...")
    sample = df_filtered.head(20)[['Dish_Name', 'Ingredients', 'Calories_100g', 
                                     'Protein_100g', 'Fat_100g', 'Carbs_100g', 'Match_Score']]
    sample.to_csv('data_sample.csv', index=False)
    print("   ✅ Saved to 'data_sample.csv'")

if __name__ == "__main__":
    diagnose_data()