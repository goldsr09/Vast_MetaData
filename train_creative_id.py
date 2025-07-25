import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- 1. Load and Prepare Data ---
df = pd.read_csv("vast_ads-6.csv")

drop_cols = [
    'title', 'duration', 'clickthrough', 'media_urls', 'created_at',
    'adomain', 'creative_hash', 'ssai_creative_id', 'ssaicreative_id',
    'ad_xml', 'ad_metadata_json', 'vast_url', 'initial_metadata_json'
]
df = df.drop(columns=drop_cols, errors='ignore')
df = df.dropna(subset=['creative_id'])

# Target encoding
creative_encoder = LabelEncoder()
df['creative_id'] = creative_encoder.fit_transform(df['creative_id'])
num_classes = len(creative_encoder.classes_)

# Separate numeric and categorical columns
categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'creative_id']
numeric_cols = [c for c in df.columns if c not in categorical_cols + ['creative_id']]

# Label encode categorical columns
cat_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    cat_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Prepare feature arrays
X_cat = df[categorical_cols].values.astype(np.int64) if categorical_cols else None
X_num = df[numeric_cols].values.astype(np.float32) if numeric_cols else None
y = df['creative_id'].values

# Train/val/test split
X_train_cat, X_temp_cat, y_train, y_temp = train_test_split(
    X_cat, y, test_size=0.3, random_state=42
)
X_val_cat, X_test_cat, y_val, y_test = train_test_split(
    X_temp_cat, y_temp, test_size=0.5, random_state=42
)

if X_num is not None:
    X_train_num, X_temp_num = train_test_split(X_num, test_size=0.3, random_state=42)
    X_val_num, X_test_num = train_test_split(X_temp_num, test_size=0.5, random_state=42)
else:
    X_train_num = X_val_num = X_test_num = None

# --- 2. Dataset Class ---
class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat is not None else None
        self.X_num = torch.tensor(X_num, dtype=torch.float32) if X_num is not None else None
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        cat = self.X_cat[idx] if self.X_cat is not None else None
        num = self.X_num[idx] if self.X_num is not None else None
        return cat, num, self.y[idx]

train_dataset = TabularDataset(X_train_cat, X_train_num, y_train)
val_dataset = TabularDataset(X_val_cat, X_val_num, y_val)
test_dataset = TabularDataset(X_test_cat, X_test_num, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# --- 3. Transformer Model (Smaller + Regularized) ---
class TransformerTabular(nn.Module):
    def __init__(self, cat_cardinalities, num_features, d_model=64, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in cat_cardinalities
        ])
        cat_emb_dim = sum(emb.embedding_dim for emb in self.cat_embeddings)

        # Linear layer for numeric features
        self.num_linear = nn.Linear(num_features, d_model // 2) if num_features > 0 else None
        total_features = cat_emb_dim + (d_model // 2 if num_features > 0 else 0)
        self.input_linear = nn.Linear(total_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_cat, x_num):
        if x_cat is not None:
            cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_embs = torch.cat(cat_embs, dim=1)
        else:
            cat_embs = torch.zeros((x_num.size(0), 0), device=x_num.device)

        num_emb = self.num_linear(x_num) if self.num_linear is not None else torch.zeros((x_cat.size(0), 0), device=x_cat.device)
        x = torch.cat([cat_embs, num_emb], dim=1)

        x = self.input_linear(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

cat_cardinalities = [len(cat_encoders[col].classes_) for col in categorical_cols]
num_features = len(numeric_cols)
model = TransformerTabular(cat_cardinalities, num_features, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. Training with Early Stopping ---
best_val_loss = float('inf')
patience = 5
patience_counter = 0
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for cat_batch, num_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(cat_batch, num_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for cat_batch, num_batch, y_batch in val_loader:
            outputs = model(cat_batch, num_batch)
            val_loss += criterion(outputs, y_batch).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "creative_model_best.pth")
        print("  ** Best model saved **")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# --- 5. Evaluate on Test Set ---
model.load_state_dict(torch.load("creative_model_best.pth"))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for cat_batch, num_batch, y_batch in test_loader:
        preds = model(cat_batch, num_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --- 6. Save Encoders and Scaler ---
joblib.dump(creative_encoder, "creative_encoder.pkl")
joblib.dump(cat_encoders, "cat_encoders.pkl")
joblib.dump(scaler, "numeric_scaler.pkl")
joblib.dump((categorical_cols, numeric_cols), "features.pkl")
print("Model and encoders saved.")
