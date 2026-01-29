import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Initialize the tokenizer and model only once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def process_mask_with_text_token(text_token, conditional_mask):
    # Function to encode text tokens to embeddings
    def encode_texts(text_list):
        encoded_inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        # Get the embeddings from the last hidden state (CLS token representation)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    # Encode the given text token
    embeddings = encode_texts([text_token])

    # Function to expand and merge embeddings with the mask
    def expand_and_merge(embedding, mask):
        # Assume mask shape is [1, 96, 96, 64], i.e., no channel dimension
        expanded_embedding = embedding.view(1, -1, 1, 1, 1).expand(-1, -1, *mask.shape[1:])  # Adapted to match spatial dimensions
        # Insert a new channel dimension for concatenation
        mask_with_channel = mask.unsqueeze(1)  # Now mask shape is [1, 1, 96, 96, 64]
        # Concatenate along the new channel dimension
        merged = torch.cat([mask_with_channel, expanded_embedding], dim=1)  # Size: [1, 769, 96, 96, 64]
        # Convolution to mix and reduce dimensions
        conv = nn.Conv3d(in_channels=769, out_channels=1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # Apply convolution to merged tensor
        final_output = conv(merged).squeeze(1)  # Remove channel dimension, shape becomes [1, 96, 96, 64]
        return final_output

    # Process the mask with the embeddings
    conditioned_mask = expand_and_merge(embeddings, conditional_mask)
    return conditioned_mask

# Example usage
text_token = "mask of tumor"
conditional_mask = torch.rand(1, 96, 96, 64)  # Example mask with no initial channel dimension
processed_mask = process_mask_with_text_token(text_token, conditional_mask)

print(processed_mask.shape)  # Should print torch.Size([1, 96, 96, 64])