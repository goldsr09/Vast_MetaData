import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

# --- Load Model and Encoders ---
creative_encoder = joblib.load("creative_encoder.pkl")
categorical_cols, feature_cols = joblib.load("features.pkl")

class TransformerModel(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Load model
num_classes = len(creative_encoder.classes_)
num_features = len(feature_cols)
model = TransformerModel(num_features=num_features, num_classes=num_classes)
model.load_state_dict(torch.load("creative_model.pth"))
model.eval()

# --- Function to Predict ---
def predict_creative(new_data_dict):
    """
    new_data_dict = {'call_number': 1, 'ad_id': '12345', 'channel_name': 'some_channel', 'wrapped_ad': 0}
    """
    data = pd.DataFrame([new_data_dict])
    # Fill missing columns
    for col in feature_cols:
        if col not in data:
            data[col] = 0
    data = data[feature_cols]

    # Encode categorical columns
    for col in categorical_cols:
        data[col] = data[col].astype(str).astype("category").cat.codes

    x = torch.tensor(data.values, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()
    return creative_encoder.inverse_transform([pred])[0]

# --- Example Usage ---
if __name__ == "__main__":
    sample_input = {
        "call_number": 1,
        "ad_id": "86422190.139962337348864",
        "channel_name": "noreservations",
        "wrapped_ad": 0
    }
    print("Predicted creative_id:", predict_creative(sample_input))
