from ultralytics import YOLO
import cv2
from ocr import predictor
import torch
import torch.nn as nn
from PIL import Image
from doctr.io import DocumentFile
import numpy as np
import tempfile
import time




#========== Custom Transform for Rotation Model ==========
class DocumentAngleCNN(nn.Module):
    def __init__(self):
        super(DocumentAngleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CustomTransform:
    def __init__(self, resize=(256, 256), normalize=True):
        self.resize = resize
        self.normalize = normalize

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if self.normalize:
            image = (image / 255.0 - 0.5) * 2.0
        return image

def predict_angle(model, image, transform, device="cpu"):
    model.eval()
    image_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_t).cpu().numpy()[0]
        pred_angle_rad = np.arctan2(output[0], output[1])
        pred_angle_deg = np.rad2deg(pred_angle_rad) % 360
    return pred_angle_deg


# =========== models loading ===========
model_dhl = YOLO(r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\best_yolo_weights_for_dhl_label.pt")

model_rot = DocumentAngleCNN()
model_rot.load_state_dict(torch.load(
    r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\foundry_rotation_model2.pth",
    map_location=torch.device('cpu'))
)

model_text = YOLO(r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\foundry_yolo_for_textfeld.pt")


start_time = time.time()
# ========== Step 1: DHL Label Detection ==========
# model_dhl = YOLO(r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\best_yolo_weights_for_dhl_label.pt")
image_path = r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\testen\input\test\122220016326836954977040-2025228173218.jpg"

# Load image once (in memory)
image = cv2.imread(image_path)
results = model_dhl.predict(source=image, conf=0.25, iou=0.45, device="cpu")

# Extract cropped DHL label region directly in RAM
# Assuming zoom_into_bbox can accept arrays — if not, modify it to do so.
bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
cropped_dhl_labels = []

for x1, y1, x2, y2 in bboxes:
    cropped = image[y1:y2, x1:x2]
    cropped_dhl_labels.append(cropped)

# Use the first detected label
if not cropped_dhl_labels:
    raise ValueError("No DHL label detected!")
label_img = cropped_dhl_labels[0]

# ========== Step 2: Rotation Correction ==========
transform = CustomTransform(resize=(256, 256), normalize=True)
predicted_angle = predict_angle(model_rot, label_img, transform, device="cpu")

# Rotate in memory
pil_img = Image.fromarray(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB))
rotated_pil = pil_img.rotate(-predicted_angle, expand=True)
rotated_img = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)

# ========== Step 3: Text Field Detection ==========
results_text = model_text.predict(source=rotated_img, conf=0.25, iou=0.45, device="cpu")

bboxes_text = results_text[0].boxes.xyxy.cpu().numpy().astype(int)
cropped_text_fields = [rotated_img[y1:y2, x1:x2] for x1, y1, x2, y2 in bboxes_text]

if not cropped_text_fields:
    raise ValueError("No text field detected!")
text_field_img = cropped_text_fields[0]

# ========== Step 4: OCR directly from memory ==========
# Write to a temporary JPEG file for Doctr
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    cv2.imwrite(tmp.name, cv2.cvtColor(text_field_img, cv2.COLOR_RGB2BGR))
    temp_path = tmp.name

doc = DocumentFile.from_images(temp_path)

result = predictor(doc)

# Extract postal codes
postal_code = None

for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                if len(word.value) in [4, 5]:
                    try:
                        int(word.value)  # Check if numeric
                        postal_code = word.value  # Save it (will overwrite previous)
                    except ValueError:
                        pass

if postal_code:
    print("The postal code:", postal_code)


end_time = time.time()

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
