import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("./sentimentdataset.csv")

# Define text preprocessing function with improved cleaning
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # More comprehensive cleaning
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content without # symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    # More advanced tokenization and lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)

# Preprocess text data
df["ProcessedText"] = df["Text"].apply(preprocess_text)

# Extract hashtags as a feature
def extract_hashtags(hashtag_str):
    if pd.isna(hashtag_str) or hashtag_str == '':
        return []
    return hashtag_str.split()

df['HashtagsList'] = df['Hashtags'].apply(extract_hashtags)
df['HashtagCount'] = df['HashtagsList'].apply(len)

# Feature engineering from datetime
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # If timestamp conversion failed, try to use the individual date/time columns
    if df['Timestamp'].isna().any() and all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
        df['Timestamp'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour']].fillna(0).astype(int),
            format='%Y%m%d%H',
            errors='coerce'
        )
    
    # Extract time features
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 5,6 = weekend
    df['TimeOfDay'] = df['Hour'].apply(lambda h: 
                                     'Morning' if 5 <= h < 12 else
                                     'Afternoon' if 12 <= h < 17 else
                                     'Evening' if 17 <= h < 21 else
                                     'Night')

# Process engagement metrics
engagement_cols = ['Retweets', 'Likes']
for col in engagement_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Feature: Engagement ratio
if all(col in df.columns for col in engagement_cols):
    df['EngagementTotal'] = df['Retweets'] + df['Likes']
    df['RetweetRatio'] = df.apply(lambda row: row['Retweets'] / row['EngagementTotal'] 
                                if row['EngagementTotal'] > 0 else 0, axis=1)

# Feature: Text properties
df['TextLength'] = df['ProcessedText'].apply(len)
df['WordCount'] = df['ProcessedText'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
df['AvgWordLength'] = df.apply(lambda row: np.mean([len(word) for word in row['ProcessedText'].split()]) 
                              if row['WordCount'] > 0 else 0, axis=1)

# Print dataset info
print(f"Dataset shape: {df.shape}")
print(f"Number of unique sentiments: {df['Sentiment'].nunique()}")
print(f"Class distribution (top 10): \n{df['Sentiment'].value_counts().head(10)}")

# Remove empty text
df = df[df['ProcessedText'].str.strip() != '']

# Group similar sentiments to reduce class count
# This is a key improvement to handle the high number of classes
sentiment_groups = {
    'Positive': ['Positive', 'Happy', 'Excited', 'Joy', 'Satisfied', 'Grateful', 'Hopeful', 
                'Optimistic', 'Pleased', 'Cheerful', 'Enthusiastic', 'Delighted', 'Good', 'Great'],
    'Negative': ['Negative', 'Sad', 'Angry', 'Frustrated', 'Disappointed', 'Worried', 'Anxious',
                'Annoyed', 'Upset', 'Concerned', 'Unhappy', 'Distressed', 'Irritated', 'Bad'],
    'Neutral': ['Neutral', 'Indifferent', 'Calm', 'Balanced', 'Objective', 'Normal'],
    'Surprised': ['Surprised', 'Shocked', 'Amazed', 'Astonished'],
    'Confused': ['Confused', 'Uncertain', 'Puzzled', 'Perplexed'],
    'Fear': ['Fear', 'Scared', 'Afraid', 'Terrified', 'Nervous'],
    'Disgust': ['Disgust', 'Repulsed', 'Revolted']
}

# Create a mapping function
def map_sentiment(sentiment):
    for group, values in sentiment_groups.items():
        if sentiment.lower() in [v.lower() for v in values]:
            return group
    return sentiment  # Keep original if not in any group, changed from 'Other'

# Apply mapping to reduce number of classes
df['GroupedSentiment'] = df['Sentiment'].apply(map_sentiment)
print(f"Number of unique sentiments after grouping: {df['GroupedSentiment'].nunique()}")

# Filter classes with few samples (optional)
min_samples_per_class = 5
class_counts = df["GroupedSentiment"].value_counts()
valid_classes = class_counts[class_counts >= min_samples_per_class].index
df = df[df["GroupedSentiment"].isin(valid_classes)]
print(f"Classes remaining after filtering: {df['GroupedSentiment'].nunique()}")
print(f"Distribution after filtering: \n{df['GroupedSentiment'].value_counts()}")

# IMPORTANT: Encode labels AFTER filtering
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["GroupedSentiment"])
print(f"Number of samples after filtering: {len(df)}")
print(f"Number of unique sentiments after filtering: {len(np.unique(y))}")

# Define features to use
text_feature = 'ProcessedText'
categorical_features = ['Platform', 'Country', 'TimeOfDay']
numerical_features = ['HashtagCount', 'IsWeekend', 'TextLength', 'WordCount', 'AvgWordLength']

# Filter out any features not in the dataframe
categorical_features = [f for f in categorical_features if f in df.columns]
numerical_features = [f for f in numerical_features if f in df.columns]

print("Using features:")
print(f"- Text feature: {text_feature}")
print(f"- Categorical features: {categorical_features}")
print(f"- Numerical features: {numerical_features}")

# Fill missing values in feature columns
for feat in categorical_features:
    df[feat] = df[feat].fillna('unknown')
for feat in numerical_features:
    df[feat] = df[feat].fillna(0)

# Split dataset AFTER encoding labels
X_text = df[text_feature]
X_categorical = df[categorical_features] if categorical_features else pd.DataFrame()
X_numerical = df[numerical_features] if numerical_features else pd.DataFrame()

# Use stratified split if possible, otherwise use regular split
try:
    X_text_train, X_text_test, X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_categorical, X_numerical, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Using stratified split")
except ValueError:
    print("Warning: Stratified split not possible. Using random split.")
    X_text_train, X_text_test, X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_categorical, X_numerical, y, test_size=0.2, random_state=42
    )

print(f"Train set shape: {len(X_text_train)} samples")
print(f"Test set shape: {len(X_text_test)} samples")

# Process text features with better TF-IDF parameters
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.9,
    sublinear_tf=True  # Apply sublinear tf scaling (logarithmic)
)
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

# Use mutual_info_classif instead of chi2 for better feature selection
selector = SelectKBest(mutual_info_classif, k=min(3000, X_text_train_tfidf.shape[1]))
X_text_train_selected = selector.fit_transform(X_text_train_tfidf, y_train)
X_text_test_selected = selector.transform(X_text_test_tfidf)
print(f"Text features after selection: {X_text_train_selected.shape[1]}")

# Process categorical features
if not X_categorical.empty:
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_train_encoded = categorical_encoder.fit_transform(X_cat_train)
    X_cat_test_encoded = categorical_encoder.transform(X_cat_test)
    print(f"Categorical features shape: {X_cat_train_encoded.shape}")
else:
    X_cat_train_encoded = np.empty((X_text_train.shape[0], 0))
    X_cat_test_encoded = np.empty((X_text_test.shape[0], 0))

# Process numerical features
if not X_numerical.empty:
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    print(f"Numerical features shape: {X_num_train_scaled.shape}")
else:
    X_num_train_scaled = np.empty((X_text_train.shape[0], 0))
    X_num_test_scaled = np.empty((X_text_test.shape[0], 0))

# Combine all features
X_train_combined = np.hstack((X_text_train_selected.toarray(), X_cat_train_encoded, X_num_train_scaled))
X_test_combined = np.hstack((X_text_test_selected.toarray(), X_cat_test_encoded, X_num_test_scaled))
print(f"Combined features shape: {X_train_combined.shape}")

# Apply class weights to handle imbalanced dataset
class_counts = Counter(y_train)
total_samples = len(y_train)
class_weights = {class_idx: total_samples / (len(class_counts) * count) for class_idx, count in class_counts.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.FloatTensor([class_weights.get(i, 1.0) for i in range(len(np.unique(y)))]).to(device)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_combined)
X_test_tensor = torch.FloatTensor(X_test_combined)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Increase batch size if classes are reduced
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the improved model with better architecture
class ImprovedSentimentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(ImprovedSentimentClassifier, self).__init__()
        
        # Improved architecture with better capacity control
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Use ELU activation for better gradient flow
        x = torch.nn.functional.elu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = torch.nn.functional.elu(self.layer2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = torch.nn.functional.elu(self.layer3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        return self.output_layer(x)

# Set device
print(f"Using device: {device}")

# Initialize model, loss function, and optimizer
input_dim = X_train_combined.shape[1]
num_classes = len(np.unique(y))
print(f"Building model with {input_dim} input features and {num_classes} output classes")

model = ImprovedSentimentClassifier(input_dim, num_classes).to(device)

# Use weighted cross entropy loss to handle imbalanced classes
criterion = nn.CrossEntropyLoss(weight=weights)

# Use AdamW optimizer with weight decay for better regularization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler with cosine annealing for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels

# Training loop with early stopping
num_epochs = 50
best_accuracy = 0.0
patience = 7
no_improvement = 0

try:
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model and check for early stopping
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'label_encoder': label_encoder,
                'vectorizer': vectorizer,
                'categorical_encoder': categorical_encoder if not X_categorical.empty else None,
                'scaler': scaler if not X_numerical.empty else None,
                'categorical_features': categorical_features,
                'numerical_features': numerical_features,
                'selector': selector,
            }, 'best_sentiment_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation accuracy: {val_acc:.4f}')
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break

except KeyboardInterrupt:
    print("Training interrupted!")

# Load the best model for final evaluation
try:
    checkpoint = torch.load('best_sentiment_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['accuracy']:.4f}")
except FileNotFoundError:
    print("Best model file not found. Continuing with current model state.")

# Final evaluation on test set
test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=1))