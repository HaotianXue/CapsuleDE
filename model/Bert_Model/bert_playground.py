"""
The playground of using pre-trained BERT model

Author: Haotian Xue
"""

import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "Who was Jim Henson?"
input_ids = tokenizer.encode(text, add_special_tokens=True)

texts = "I love you"
input_id = tokenizer.encode_plus(texts, add_special_tokens=True, max_length=10, pad_to_max_length=True)

print(input_id)

input_ids = input_id['input_ids']
attn_msk = torch.tensor([input_id['attention_mask']])
print("attention mask: ", attn_msk.shape)  # (batch, sen_len)


# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([input_ids])
print(tokens_tensor)
print(tokens_tensor.shape)  # (batch, sen_len)

model = BertModel.from_pretrained('bert-base-cased')
"""
for p in model.parameters():
    print(p)
"""

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, attn_msk)
    # print(outputs)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]  # hidden_size=768
    print(encoded_layers.shape)
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
# assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

"""
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor)
    logits = outputs[0]
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    print(logits)
"""

