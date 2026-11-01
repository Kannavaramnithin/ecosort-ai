import torch
from torchvision import models, transforms
from PIL import Image
MODEL_PATH = r"F:\ECOSORTAI\ECO\best_model.pth"
IMAGE_PATH = r"F:\ECOSORTAI\ECO\waste_item.jpg"

# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = r"F:\ECOSORTAI\ECO\best_model.pth"


# ==========================================
# CLASS NAMES
# ==========================================

classes = [
    "metal",
    "glass",
    "biological",
    "paper",
    "battery",
    "trash",
    "cardboard",
    "shoes",
    "clothes",
    "plastic"
]


# ==========================================
# CREATE RESNET101
# ==========================================

print("Creating model...")

model = models.resnet50(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    10
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# TEST IMAGE
# ==========================================

IMAGE_PATH = r"F:\ECOSORTAI\ECO\waste_item.jpg"

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(
    image
).unsqueeze(0)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )


# ==========================================
# SHOW ALL PREDICTIONS
# ==========================================

print()
print("==============================")
print("       ECOSORT AI")
print("==============================")

for i, probability in enumerate(probabilities[0]):

    print(
        f"{classes[i]:15s} : "
        f"{probability.item() * 100:.2f}%"
    )


# ==========================================
# FINAL PREDICTION
# ==========================================

predicted_index = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = (
    probabilities[0][predicted_index].item()
    * 100
)

print("==============================")

print(
    "Prediction :",
    classes[predicted_index]
)

print(
    "Confidence :",
    f"{confidence:.2f}%"
)

print("==============================")