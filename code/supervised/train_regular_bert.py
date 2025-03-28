import pandas as pd
from sklearn.model_selection import train_test_split

# Import the main function from bert_classification_functions.py
from classification.bert_classification_functions import train_classifier1

# 1) Read the dataset from a CSV file
df = pd.read_csv('.csv')

# 2) Remove rows with NaN values in the relevant columns
df = df.dropna(subset=['Summary_Description_Cleaned', 'label'])

# 3) Convert string labels to numeric values
df['label'] = df['label'].map({'ATD': 1, 'Non-ATD': 0})

# 4) Split text and label into separate variables
all_text = df['Summary_Description_Cleaned'].tolist()
all_label = df['label'].tolist()

# 5) Split the dataset into training and testing sets
train_text, test_text, train_label, test_label = train_test_split(
    all_text, 
    all_label, 
    test_size=0.2, 
    random_state=42
)

# 6) Define the directory for saving the model/results
output_filepath = '/scratch/p311371/bert_results/'

# 7) Call the train_classifier1 function
trainer, evaluation = train_classifier1(
    xtrain=train_text, 
    ytrain=train_label,
    xtest=test_text, 
    ytest=test_label,
    output_filepath=output_filepath,
    epochs=50,          
    batch_size=16,       
    max_length=512,      
    model_name='bert-base-uncased'
)

# 8) Print the evaluation summary
print("Evaluation Results:")
print(evaluation)
