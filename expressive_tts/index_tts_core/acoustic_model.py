"""
Instruction to user:
This is a mock version of what an Index-TTS core modification looks like.
You must update your own model classes (like the AcousticModel) to incorporate this properly.
"""
import torch
import torch.nn as nn

class ModifiedAcousticModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_emotions=7):
        super().__init__()
        
        self.phoneme_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 3. Add Emotion Embedding Injection
        self.emotion_embedding = nn.Embedding(num_emotions, 128)
        
        # Adjust encoder input dim
        self.encoder_input_proj = nn.Linear(hidden_dim + 128, hidden_dim)
        
        self.encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # Note: You need to plug this into your real encoder
        
    def forward(self, phoneme_seq, emotion_id, positional_encoding=None):
        """
        phoneme_seq: (B, T)
        emotion_id: (B)
        """
        B, T = phoneme_seq.shape
        
        # (B, T, hidden_dim)
        phone_emb = self.phoneme_embedding(phoneme_seq)
        
        # (B, 128) -> (B, 1, 128) -> (B, T, 128)
        emo_emb = self.emotion_embedding(emotion_id)
        emo_emb = emo_emb.unsqueeze(1).expand(-1, T, -1)
        
        if positional_encoding is None:
            positional_encoding = torch.zeros_like(phone_emb)
            
        # Concatenate phoneme_embedding + emotion_embedding + positional_encoding
        encoder_input_concat = torch.cat([phone_emb, emo_emb], dim=-1)
        
        # Project down to hidden dim
        encoder_input = self.encoder_input_proj(encoder_input_concat) + positional_encoding
        
        # Encoder forward
        out, _ = self.encoder(encoder_input)
        
        return out

if __name__ == "__main__":
    # Test
    model = ModifiedAcousticModel(vocab_size=50)
    print("Model initialized with emotion embeddings!")
    p_seq = torch.randint(0, 50, (1, 10))
    e_id = torch.tensor([0])
    out = model(p_seq, e_id)
    print("Encoder output shape:", out.shape)
