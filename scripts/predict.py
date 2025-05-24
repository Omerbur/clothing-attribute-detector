from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# 💡 Attributes your model was trained on
ATTRIBUTES = ['Red', 'Blue', 'Striped', 'Solid', 'LongSleeve', 'ShortSleeve']

# 🧠 Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, len(ATTRIBUTES)),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
model.eval()

# 📸 Image loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔍 Prediction
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        preds = output.squeeze().numpy()
    results = {ATTRIBUTES[i]: bool(preds[i] > 0.5) for i in range(len(ATTRIBUTES))}
    return results

# 🧪 Example usage
if __name__ == "__main__":
    img_path = "sample.jpg"  # Replace with your test image path
    result = predict(img_path)
    print("Predicted Attributes:", result)
