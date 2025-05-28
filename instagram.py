import pandas as pd

def combine_csv(train_path='train.csv', test_path='test.csv', output_path='instagram.csv'):
    """Combine train.csv and test.csv into instagram.csv"""
    try:
        # Load the datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Verify that columns match
        if list(train_df.columns) != list(test_df.columns):
            raise ValueError("Column mismatch between train.csv and test.csv. Ensure both files have the same columns.")
        
        # Combine datasets
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Check for duplicates (optional)
        if combined_df.duplicated().sum() > 0:
            print(f"Warning: {combined_df.duplicated().sum()} duplicate rows found. Removing duplicates...")
            combined_df = combined_df.drop_duplicates()
        
        # Save to instagram.csv
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(combined_df)} rows")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure train.csv and test.csv are in the project directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    combine_csv()