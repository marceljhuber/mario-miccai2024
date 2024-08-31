import pandas as pd
import numpy as np

# Load the predictions CSV
predictions_csv = 'validation_predictions.csv'
df = pd.read_csv(predictions_csv)

# Apply softmax function row-wise
def softmax(x):
    # Subtract max per row for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

# Function to process predictions
def process_predictions(row):
    predictions = softmax(np.array(eval(row['k_fold_predictions'])))  # Convert string to numpy array
    sum_predictions = predictions.sum(axis=0)  # Sum across all folds
    return np.argmax(sum_predictions)  # Return the class with the highest sum

# Apply the processing function to each row
df['final_prediction'] = df.apply(process_predictions, axis=1)

# Prepare the submission data
submission_data = np.column_stack((df['case'], df['final_prediction']))

# Save the submission file
np.savetxt('submission_predictions.csv',
           submission_data,
           delimiter=',',
           fmt='%s',
           header='case,prediction',
           comments='')

print("Submission file 'submission_predictions.csv' has been created.")