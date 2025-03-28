import pandas as pd
from small_text.query_strategies import BreakingTies, LeastConfidence, PredictionEntropy, DiscriminativeActiveLearning, EmbeddingKMeans, ContrastiveActiveLearning
from classification.active_learning import fullpipeline  # Import the main pipeline
from sklearn.model_selection import train_test_split

# === Load dataset ===
csv_file = ".csv"
df = pd.read_csv(csv_file)

# === Encode labels (label: ATD/Non-ATD → 1/0) ===
label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
df["label"] = df["label"].map(label_mapping)

# === Extract texts and labels ===
texts = df["Summary_Description_Cleaned"].values
labels = df["label"].values

# === Perform stratified train-test split ===
train_text, test_text, train_idx, test_idx = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# === Active Learning Parameters ===
query_size = 100           # initial number of samples
num_queries = 25           # number of active learning iterations
query_strategy = PredictionEntropy()
transformer_model = "bert-base-uncased"
filepath = "./al-results/"
num_classes = 2
batch_size = 32
num_epochs = 20

# === Run Active Learning ===
labelled_df, results_df = fullpipeline(
    train_idx, train_text,
    test_idx, test_text,
    query_size, num_queries,
    query_strategy, transformer_model,
    filepath, num_classes,
    batch_size, num_epochs
)

# === Save results ===
labelled_df.to_csv(filepath + "AL_labeled_data.csv", index=False)
results_df.to_csv(filepath + "AL_active_learning_results.csv", index=False)

print("\nActive Learning Process Completed.")
