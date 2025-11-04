import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from doctr.io import DocumentFile
from ultralytics import YOLO
from ocr import predictor  # your OCR module
import torch.nn as nn

# ========== Custom Classes and Functions ==========
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


# ========== Load Models Once ==========
model_dhl = YOLO(r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\best_yolo_weights_for_dhl_label.pt")
model_text = YOLO(r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\foundry_yolo_for_textfeld.pt")

model_rot = DocumentAngleCNN()
model_rot.load_state_dict(torch.load(
    r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\foundry_modele\foundry_rotation_model2.pth",
    map_location=torch.device('cpu'))
)

transform = CustomTransform(resize=(256, 256), normalize=True)

# ========== Paths ==========
image_folder = r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\testen\input\test"
excel_path = r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\mainskript\testen\input\test\images_plz.xlsx"

# ========== Load Excel ==========
df = pd.read_excel(excel_path, dtype={"plz_should": str, "plz_is": str})

# Strip any whitespace
df["plz_should"] = df["plz_should"].str.strip()
df["plz_is"] = df["plz_is"].str.strip()

# Add empty columns if not exist
if 'plz_is' not in df.columns:
    df['plz_is'] = ""
if 'runtime' not in df.columns:
    df['runtime'] = 0.0


# ========== Process Each Image ==========
for idx, row in df.iterrows():
    image_name = row['image_name_without_jpg_suffix'] + ".jpg"
    image_path = os.path.join(image_folder, image_name)

    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_name}")
        continue

    print(f"\n🔍 Processing: {image_name}")
    start_time = time.time()

    try:
        # Step 1: DHL Label Detection
        image = cv2.imread(image_path)
        results = model_dhl.predict(source=image, conf=0.25, iou=0.45, device="cpu")
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        if not len(bboxes):
            print("No DHL label detected.")
            df.loc[idx, 'plz_is'] = "NO_LABEL"
            continue
        x1, y1, x2, y2 = bboxes[0]
        label_img = image[y1:y2, x1:x2]

        # Step 2: Rotation Correction
        predicted_angle = predict_angle(model_rot, label_img, transform, device="cpu")
        pil_img = Image.fromarray(cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB))
        rotated_pil = pil_img.rotate(-predicted_angle, expand=True)
        rotated_img = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)

        # Step 3: Text Field Detection
        results_text = model_text.predict(source=rotated_img, conf=0.25, iou=0.45, device="cpu")
        bboxes_text = results_text[0].boxes.xyxy.cpu().numpy().astype(int)
        if not len(bboxes_text):
            print("No text field detected.")
            df.loc[idx, 'plz_is'] = "NO_TEXT"
            continue
        x1, y1, x2, y2 = bboxes_text[0]
        text_field_img = rotated_img[y1:y2, x1:x2]

        # Step 4: OCR
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(text_field_img, cv2.COLOR_RGB2BGR))
            temp_path = tmp.name

        doc = DocumentFile.from_images(temp_path)
        result = predictor(doc)

        # Step 5: Extract Postal Code
        plz_found = None
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if len(word.value) in [4, 5]:
                            try:
                                int(word.value)  # check if numeric
                                plz_found = word.value  # store latest valid postal code
                            except ValueError:
                                continue

        df.loc[idx, 'plz_is'] = plz_found if plz_found else "NOT_FOUND"

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        df.loc[idx, 'plz_is'] = "ERROR"

    end_time = time.time()
    df.loc[idx, 'runtime'] = round(end_time - start_time, 2)

    print(f"✅ Done ({df.loc[idx, 'runtime']}s) | PLZ: {df.loc[idx, 'plz_is']}")

# ========== Save Updated Excel ==========
# output_path = excel_path.replace(".xlsx", "_updated.xlsx")
# df.to_excel(output_path, index=False)
# print(f"\n✅ All done! Updated Excel saved at:\n{output_path}")


# ========== Average runtime and accuracy ==========

# Calculate average runtime
avg_runtime = df["runtime"].mean()

# Calculate accuracy: proportion where plz_is == plz_should
count = (df["plz_is"] == df["plz_should"]).sum()

# Accuracy using len(df)-1
accuracy = count / (len(df))
print("len(df):", len(df))

# Just print the results
print(f"\n📊 Average runtime: {avg_runtime:.2f} seconds")
print(f"📈 Average accuracy: {accuracy:.2%}")

#  Add them as new columns in the Excel output
df["avg_runtime"] = avg_runtime
df["avg_accuracy"] = accuracy

# Save updated Excel file again
output_path_summary = excel_path.replace(".xlsx", "_updated_with_summary.xlsx")
df.to_excel(output_path_summary, index=False)
print(f"✅ Summary saved at:\n{output_path_summary}")
