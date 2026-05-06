import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from huggingface_hub import login

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import losses, datasets
from torch.utils.data import DataLoader
import nltk

hf_token = None

segments_filepath = '../data/segments/merged_tibetan_segments.xlsx'
full_segmentation_with_noise_scores = '../data/segments/merged_tibetan_segments_v2_with_noise_scores.xlsx'
new_model_name = 'tibetan-tsdae-base'
model_name = 'OMRIDRORI/mbert-tibetan-continual-wylie-final'
model_v2_name = 'Intellexus/IntellexusBert-2.0'


if hf_token:
    login(token=hf_token)


class NoiseCriterion:
    def __init__(self, model_name=model_name, device='cuda'):
        """
        Initializes the BERT model for Masked Language Modeling (MLM).
        """
        self.device = device
        print(f"Loading model_name: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_reconstruction_loss_score(self, sentence, batch_size=32):
        """
        Calculates P(S) with memory safety.
        Splits the workload into mini-batches to prevent OOM on long sentences.
        """
        # 1. Tokenize with truncation to ensure we don't exceed model limits (usually 512)
        inputs = self.tokenizer(sentence, return_tensors='pt',
                                add_special_tokens=True,
                                truncation=True,
                                max_length=512)

        input_ids = inputs['input_ids'].to(self.device)
        seq_len = input_ids.shape[1]

        # We mask tokens from index 1 to seq_len-1 (skipping CLS and SEP)
        target_indices = list(range(1, seq_len - 1))
        actual_length = len(target_indices)

        if actual_length == 0:
            return 0.0

        # 2. Create the massive batch of masked inputs (on CPU first to save GPU memory)
        # We repeat the input_ids N times.
        # Note: We keep this on device if possible for speed, but moving the repeat logic
        # inside the loop is even safer for memory if sentences are extreme.
        # For now, repeating on device is usually fine if we process in chunks.
        batched_input_ids = input_ids.repeat(actual_length, 1)

        # Apply diagonal masking
        mask_token_id = self.tokenizer.mask_token_id
        indices_tensor = torch.tensor(target_indices, device=self.device)
        batched_input_ids[torch.arange(actual_length), indices_tensor] = mask_token_id

        total_loss = 0.0

        # 3. Process in Mini-Batches (The Fix)
        # Instead of self.model(batched_input_ids), we loop.
        for i in range(0, actual_length, batch_size):
            # Slice a mini-batch
            batch_end = min(i + batch_size, actual_length)
            mini_batch_input = batched_input_ids[i:batch_end]

            with torch.no_grad():
                outputs = self.model(mini_batch_input)
                predictions = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Calculate loss for the specific masked tokens in this mini-batch
            for local_idx in range(len(mini_batch_input)):
                # The global index in the sequence of target_indices
                global_idx = i + local_idx
                token_pos = target_indices[global_idx]

                # Get logits for the specific masked position
                token_logits = predictions[local_idx, token_pos, :]
                true_token_id = input_ids[0, token_pos]

                loss = F.cross_entropy(token_logits.view(1, -1), true_token_id.view(-1))
                total_loss += loss.item()

            # clear cache explicitly to be safe
            del outputs, predictions, mini_batch_input

        # Cleanup
        del batched_input_ids, input_ids, inputs

        if total_loss == 0:
            return float('inf')

        p_score = actual_length / total_loss
        return p_score

    def compute_noise_rank(self, sentence_a, sentence_b):
        p_a = self.get_reconstruction_loss_score(sentence_a)
        p_b = self.get_reconstruction_loss_score(sentence_b)
        return -p_a - p_b



def tsdae():
    #nltk.download('punkt', '.venv/nltk_data')
    #nltk.download('punkt_tab', '.venv/nltk_data')

    # 1. Pick a base model (Multilingual BERT or similar)
    model = SentenceTransformer(model_name, torch_dtype=torch.bfloat16, device='cuda')

    segments_df = pd.read_excel(full_segmentation_with_noise_scores)
    tibetan_sentences = segments_df['Segmented_Text_EWTS'].tolist()

    # 2. Load your raw Tibetan text (One sentence per line)
    # No need for pairs! Just a raw list.
    # tibetan_sentences = [
    #     "chos thams cad sems kyi rnam 'phrul yin",
    #     "sems nyid 'od gsal ba'i rang bzhin no",
    #     # ... load 100k+ lines here
    # ]

    # 3. Create the Special Denoising Dataset
    # This automatically handles the "Delete 60% of words" logic
    train_dataset = datasets.DenoisingAutoEncoderDataset(tibetan_sentences)

    # 4. DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 5. The Loss Function (The Magic)
    # This attaches a Decoder to your Encoder during training
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=model_name,
        tie_encoder_decoder=True  # Important: Share weights between encoder/decoder
    )

    # 6. Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    # 7. Result
    # Now you have a model adapted to Tibetan without a single labeled pair!
    model.save(path=new_model_name)

def eval_noise():
    print(f"Loading segments file: {full_segmentation_with_noise_scores} ...")
    df = pd.read_excel(full_segmentation_with_noise_scores)

    df_clean = df.replace([np.inf, -np.inf], np.nan)

    # 3. Calculate the mean (pandas automatically ignores NaNs)
    average_noise = df_clean['noise_score'].mean()

    print(f"Average Noise Score: {average_noise}")

    # Optional: Check how many scores were infinite
    inf_count = len(df) - len(df_clean.dropna(subset=['noise_score']))
    print(f"Number of perfect/infinite scores skipped: {inf_count}")

def main(model_name=model_name, segmentation_with_noise_scores=full_segmentation_with_noise_scores):

    print(f"Loading segments file: {segments_filepath} ...")
    segments_df = pd.read_excel(segments_filepath)

    noise_scorer = NoiseCriterion(device='cuda', model_name=model_name)  # Use 'cuda' if GPU available

    print(f"Iterating on segments: {len(segments_df)} ...")
    segments_length = len(segments_df)
    pbar = tqdm(segments_df.iterrows(), total=segments_length)
    for index, row in pbar:
        sentence_a = row['Segmented_Text_EWTS']

        score = noise_scorer.get_reconstruction_loss_score(sentence_a)
        segments_df.at[index, 'noise_score'] = score
        pbar.set_description(f"Processed index {index}: Score = {score:.4f}")

        # Inside main loop
        if index % 100 == 0:
            torch.cuda.empty_cache()

        if index > 0 and index % 10000 == 0:
            df_clean = segments_df.replace([np.inf, -np.inf], np.nan)
            average_noise = df_clean['noise_score'].mean()
            print(f"Average Noise for index {index}. Score: {average_noise}")

            # Save the updated DataFrame with noise scores
            segments_df.to_excel(full_segmentation_with_noise_scores, index=False)
            break

    # Save the updated DataFrame with noise scores
    segments_df.to_excel(full_segmentation_with_noise_scores, index=False)
    print("Completed processing all segments and saved noise scores.")


# --- Usage Example ---
if __name__ == "__main__":
    main(model_name=model_v2_name, segmentation_with_noise_scores=full_segmentation_with_noise_scores)
    eval_noise()
    #tsdae()


# 2016.02.01 Goody new segments.
# Average Noise Score: 0.09369212186205825


# T v1 model
# Average Noise for index 30000. Score: 0.28704316535802993

# T v2 model
# Average Noise Score: 0.07858142938638263