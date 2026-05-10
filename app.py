import os
import gdown

# --- 2. WAKE UP THE AI BRAIN ---
print("Checking for AI Model...")

model_path = 'skin_tone_model.pth'

# If the model isn't on the computer, download it from Google Drive!
if not os.path.exists(model_path):
    print("Model not found locally. Downloading from the cloud (this takes a minute)...")
    # PASTE YOUR GOOGLE DRIVE FILE ID HERE:
    file_id ="1E0WXY1fBa6FhNr1yJs5yY0-auMKs7KYl"
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)
else:
    print("Model found locally!")

print("Loading AI Model into memory...")
# Rebuild the empty brain structure
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

# Load your specific memories into the brain
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set to testing mode
    print("✅ AI Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
