import pandas as pd

# Load training data
train_data_path = "train_data.df"
train_data = pd.read_pickle(train_data_path)

# Reduce dataset (e.g., keep 1000 samples)
train_data_small = train_data.sample(n=1000, random_state=42)

# Save as a new file
new_train_data_path = "train_data_small.df"
train_data_small.to_pickle(new_train_data_path)

print(f"New training data size: {len(train_data_small)}")
print(f"Saved reduced dataset to: {new_train_data_path}")
