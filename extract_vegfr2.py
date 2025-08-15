import pandas as pd

def extract_vegfr2_data(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df_vegfr2 = df[["smiles", "VEGFR2"]].dropna()
    df_vegfr2.to_csv(output_path, index=False)

if __name__ == "__main__":
    extract_vegfr2_data("datasets/train_clean.csv", "datasets/vegfr2_train.csv")
    extract_vegfr2_data("datasets/val_clean.csv", "datasets/vegfr2_val.csv") 
    extract_vegfr2_data("datasets/test_clean.csv", "datasets/vegfr2_test.csv")