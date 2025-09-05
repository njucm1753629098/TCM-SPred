import torch
import torch.nn as nn
from config import DEVICE

class EnhancedHerbModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Apply deterministic initialization
        self._initialize_weights()
        
        # Main branch
        self.main_processor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Bidirectional encoders with deterministic initialization
        self.herb_encoder = nn.LSTM(
            input_size=129,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self._init_lstm(self.herb_encoder)
        
        self.symptom_encoder = nn.LSTM(
            input_size=129,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        self._init_lstm(self.symptom_encoder)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )
        self._init_mha(self.cross_attention)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self._init_classifier(self.classifier)
    
    def _initialize_weights(self):
        """Initialize weights deterministically"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _init_lstm(self, lstm):
        """Initialize LSTM weights deterministically"""
        for name, param in lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def _init_mha(self, mha):
        """Initialize multi-head attention weights deterministically"""
        for p in mha.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _init_classifier(self, classifier):
        """Initialize classifier weights deterministically"""
        for module in classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Main feature
        main_feature = torch.cat([x['herb'], x['symptom']], dim=1)
        main_processed = self.main_processor(main_feature)

        # Herb encoding
        herb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x['herb_symbols'],
            x['herb_lengths'].cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        herb_out, _ = self.herb_encoder(herb_packed)
        herb_out, _ = torch.nn.utils.rnn.pad_packed_sequence(herb_out, batch_first=True)

        # Symptom encoding
        symptom_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x['symptom_symbols'],
            x['symptom_lengths'].cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        symptom_out, _ = self.symptom_encoder(symptom_packed)
        symptom_out, _ = torch.nn.utils.rnn.pad_packed_sequence(symptom_out, batch_first=True)

        # Pooling & interaction
        symptom_pooled = torch.mean(symptom_out, dim=1, keepdim=True)
        interaction = herb_out * symptom_pooled.expand_as(herb_out)

        # Combine features
        combined = torch.cat([
            main_processed.unsqueeze(1).expand(-1, herb_out.size(1), -1),
            herb_out,
            symptom_pooled.expand(-1, herb_out.size(1), -1),
            interaction
        ], dim=2)

        # Attention
        attn_out, _ = self.cross_attention(combined, combined, combined)

        # Global pooling
        pooled = torch.mean(attn_out, dim=1)

        # Classify
        return self.classifier(pooled)

def initialize_model():
    """Initialize model with deterministic weights"""
    model = EnhancedHerbModel().to(DEVICE)
    print(f"Model initialized on {DEVICE}")
    print(f"Model architecture:\n{model}")
    return model