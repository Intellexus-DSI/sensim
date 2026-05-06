
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
import numpy as np

import os
import pandas as pd

# --- 1. CONFIGURATION AND DUMMY DATA (MODIFIED FOR PAIRS) ---

# Use a small BERT model for speed in this example
MODEL_NAME = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
NUM_SAMPLES = 20  # Number of BWS choice sets
DATA_FILE = 'results/llms/candidate_pairs.xlsx'

MODEL_SAVE_PATH = "bws_training_model"  # Directory for HF format
MODEL_STATE_PATH = "bws_training_model/pytorch_model.bin"  # File for state dict

# Simulated BWS Data: Each row is a choice set.
# Structure: [ [Best_Sentence_A, Best_Sentence_B], [Worst_Sentence_C, Worst_Sentence_D] ]
# The model will score the concatenated string "A [SEP] B" vs "C [SEP] D".
BWS_SAMPLES = [
                  [
                      ["The new system is incredibly fast and efficient.",
                       "Its architecture provides unparalleled scalability."],
                      ["The UI is confusing and looks dated.", "The deployment process is cumbersome and error-prone."]
                  ],
                  [
                      ["The product is robust and completely bug-free.",
                       "The data visualization tools are insightful and clear."],
                      ["Customer support is unresponsive and unhelpful.",
                       "Setup requires complex manual configuration steps."]
                  ],
                  [
                      ["The security protocols are modern and compliant.",
                       "It has a flexible API for custom solutions."],
                      ["The license agreement is extremely restrictive.",
                       "Learning the advanced features takes too long."]
                  ],
                  [
                      ["Training documentation is straightforward.", "Integration with all systems is seamless."],
                      ["Performance dips significantly under high load.", "The energy consumption is quite high."]
                  ],
                  [
                      ["It offers decent security features.", "It has a large community following."],
                      ["The price point is too high for its features.", "The feature set is minimal and outdated."]
                  ],
                  # Repeat and mix samples
              ] * int(NUM_SAMPLES / 5)


# --- 2. MODEL DEFINITION (NO CHANGE NEEDED) ---

class BertForBWS(BertPreTrainedModel):
    """
    A BERT model modified to output a single utility score (logit)
    from the [CLS] token's representation. This score represents the utility
    of the *concatenated pair* input.
    """

    def __init__(self, config):
        super().__init__(config)
        # Use the standard BERT base model
        self.bert = BertModel(config)
        # A dense layer on top of the [CLS] token output to get a single score (logit)
        self.bws_score = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT output
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # We take the hidden state of the [CLS] token (the first token)
        cls_output = outputs[0][:, 0, :]

        # Pass the [CLS] representation through the scoring head
        score = self.bws_score(cls_output)

        # Score is returned as [batch_size, 1]
        return score.squeeze(-1)  # Squeeze to [batch_size] for easier loss calculation


# --- 3. BWS DATASET AND LOSS FUNCTION (DATASET MODIFIED) ---

class BWSDataFrameDataset(Dataset):
    """Dataset for BWS pairs, separating Best and Worst sentences."""

    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        row = self.samples.iloc[idx]

        best_item = row['BestItem']
        worst_item = row['WorstItem']

        best_pair = None
        worst_pair = None
        for i in range(1, 5):
            if row[f'Item{i}'] == best_item:
                best_pair = [row[f'pair_{i}_A'], row[f'pair_{i}_B']]
            elif row[f'Item{i}'] == worst_item:
                worst_pair = [row[f'pair_{i}_A'], row[f'pair_{i}_B']]

            if best_pair and worst_pair:
                break

        assert best_pair is not None
        assert worst_pair is not None

        # Tokenize the Best PAIR: Sentence A and Sentence B joined by [SEP]
        # Max length increased to 128 to accommodate two sentences and tokens.
        best_enc = self.tokenizer(
            best_pair[0],  # First sentence in the pair
            text_pair=best_pair[1],  # Second sentence in the pair
            truncation=True,
            padding='max_length',
            max_length=128*2,
            return_tensors="pt"
        )

        # Tokenize the Worst PAIR: Sentence C and Sentence D joined by [SEP]
        worst_enc = self.tokenizer(
            worst_pair[0],
            text_pair=worst_pair[1],
            truncation=True,
            padding='max_length',
            max_length=128*2,
            return_tensors="pt"
        )

        # We return the tensors for Best and Worst pairs separately
        return (
            {k: v.squeeze(0) for k, v in best_enc.items()},
            {k: v.squeeze(0) for k, v in worst_enc.items()}
        )

def bws_logit_difference_loss(v_B, v_W):
    """
    Calculates the Logit Difference Loss for the Best (v_B) and Worst (v_W) scores.
    Loss = log(1 + exp(v_W - v_B))
    This loss is minimized when v_B is much greater than v_W.
    """
    # Calculate the difference: (Worst score - Best score)
    diff = v_W - v_B

    # Use torch.log(1 + torch.exp(diff)) for numerical stability
    loss = torch.log(1 + torch.exp(diff))

    # Return the mean loss over the batch
    return torch.mean(loss)


# --- 4. TRAINING FUNCTION (NO MAJOR CHANGE NEEDED) ---

def train_bws_model():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer, model, and optimizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForBWS.from_pretrained(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Prepare data
    data_df = pd.read_csv(DATA_FILE, sep='\t', header=0)
    bws_dataset = BWSDataFrameDataset(data_df, tokenizer)
    bws_dataloader = DataLoader(bws_dataset, batch_size=4, shuffle=True)

    model.train()

    print("\nStarting Training (1 Epoch)...")

    # Tracking list to see if v_B > v_W after training
    score_diffs_before = []
    score_diffs_after = []

    # Get initial scores (Pre-training benchmark)
    model.eval()
    with torch.no_grad():
        for i, (best_data, worst_data) in enumerate(bws_dataloader):
            # Extract inputs and move to device
            best_input_ids = best_data['input_ids'].to(device)
            worst_input_ids = worst_data['input_ids'].to(device)

            # Get logits
            v_B = model(best_input_ids)
            v_W = model(worst_input_ids)

            # Store the difference
            score_diffs_before.extend((v_B - v_W).cpu().numpy())

            if i >= 5: break  # Only take a few batches for the benchmark

    model.train()
    # Actual training loop
    for epoch in range(2):  # Run for one epoch for demonstration
        total_loss = 0
        for batch_idx, (best_data, worst_data) in enumerate(bws_dataloader):
            optimizer.zero_grad()

            # --- Move data to device and get scores ---
            # Input tensors now contain the concatenated pairs
            best_input_ids = best_data['input_ids'].to(device)
            best_attention_mask = best_data['attention_mask'].to(device)
            worst_input_ids = worst_data['input_ids'].to(device)
            worst_attention_mask = worst_data['attention_mask'].to(device)

            # Get utility scores (logits)
            # The model is now scoring the utility of the combined pair
            v_B = model(best_input_ids, attention_mask=best_attention_mask)
            v_W = model(worst_input_ids, attention_mask=worst_attention_mask)

            # --- Calculate BWS Loss ---
            loss = bws_logit_difference_loss(v_B, v_W)

            # --- Backpropagation ---
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 4 == 0:
                print(f"  Batch {batch_idx}/{len(bws_dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(bws_dataloader)
        print(f"\n--- Epoch {epoch} Finished --- Average Loss: {avg_loss:.4f}")

    # Get final scores (Post-training benchmark)
    model.eval()
    with torch.no_grad():
        for best_data, worst_data in bws_dataloader:
            best_input_ids = best_data['input_ids'].to(device)
            worst_input_ids = worst_data['input_ids'].to(device)
            v_B = model(best_input_ids)
            v_W = model(worst_input_ids)
            score_diffs_after.extend((v_B - v_W).cpu().numpy())

    # --- 5. RESULTS AND ANALYSIS ---

    print("\n--- Training Results (Score Difference v_B - v_W) ---")

    # Calculate Mean Difference
    mean_diff_before = np.mean(score_diffs_before)
    mean_diff_after = np.mean(score_diffs_after)

    # Calculate 'Correct' Prediction Rate (i.e., v_B > v_W)
    correct_before = np.sum(np.array(score_diffs_before) > 0) / len(score_diffs_before)
    correct_after = np.sum(np.array(score_diffs_after) > 0) / len(score_diffs_after)

    print(f"Initial Mean Score Difference (v_B - v_W): {mean_diff_before:.4f}")
    print(f"Initial Correct Prediction Rate (v_B > v_W): {correct_before:.2%}")

    print("-" * 50)

    print(f"Final Mean Score Difference (v_B - v_W): {mean_diff_after:.4f}")
    print(f"Final Correct Prediction Rate (v_B > v_W): {correct_after:.2%}")

    if mean_diff_after > mean_diff_before:
        print("\nSuccess! The model learned to assign a higher score to the Best sentence pair.")
    else:
        print("\nWarning: The training did not improve the score difference.")

    # --- 6. SAVING THE TRAINED MODEL ---
    print("\n--- Saving Model ---")

    # Create the directory if it doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    # 1. Save in Hugging Face format (recommended for easy loading/sharing)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to Hugging Face format directory: {MODEL_SAVE_PATH}")

    # 2. Save only the PyTorch state dictionary (for checkpointing/fine-tuning)
    torch.save(model.state_dict(), MODEL_STATE_PATH)
    print(f"Model state dictionary saved to: {MODEL_STATE_PATH}")


if __name__ == '__main__':
    train_bws_model()
