from pyspark.sql import functions as F
import os
import cv2
import numpy as np
import pandas as pd
from transforms.api import transform, Input, Output, configure
from myproject.datasets.libs.utils import (
    read_dir_recursive,
)
from myproject.datasets.libs.lib_foundry_filesystem import (
    save_foundry_files_in_dir,
    write_files_to_dataset
)

import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

@configure(profile=['DRIVER_MEMORY_EXTRA_LARGE', "DRIVER_GPU_ENABLED", "KUBERNETES_NO_EXECUTORS"])
@transform(
    rotation_model=Output("/Trans-o-flex/[Use Case] MS-Daten/data/09_retraining/rotation_model/output_rotation_model2"),
    source_df=Input("/Trans-o-flex/[Use Case] MS-Daten/data/09_retraining/create_train_data_4_rotation/output/rotated_images"),
)
def compute(source_df, rotation_model):

    temp_dir = tempfile.mkdtemp()

    # Get a list of the images which should be analyzed and save them as temporary files.
    _, train_temp_dir = save_foundry_files_in_dir(source_df, temp_dir, sub_dir_name="data")
    # train_temp_dir = train_temp_dir + "/data"

    # ───────────────────────────────
    # Dataset (with sin-cos encoding)
    # ───────────────────────────────
    class RotatedImageDataset(Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.annotations = df
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
            angle_deg = self.annotations.iloc[idx, 1]
            img_name = self.annotations.iloc[idx, 0]

            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image)

            # Convert to sin-cos
            angle_rad = np.deg2rad(angle_deg)
            angle_vec = torch.tensor([np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32)

            return image, angle_vec, img_name

    # ───────────────────────────────
    # Custom Transform (OpenCV style)
    # ───────────────────────────────
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

    # ───────────────────────────────
    # CNN Model (with Dropout)
    # ───────────────────────────────
    class DocumentAngleCNN(nn.Module):
        def __init__(self):
            super(DocumentAngleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(128 * 32 * 32, 512)
            self.dropout = nn.Dropout(p=0.5)   # Regularization
            self.fc2 = nn.Linear(512, 2)       # sin & cos outputs

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

        # Step 3: Prepare the Dataset and DataLoader
    def get_data_loaders(csv_file, img_dir, batch_size=16):
        # Read CSV
        df = pd.read_csv(csv_file)

        # Filter out rows where image file does not exist
        df = df[df.iloc[:, 0].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))].reset_index(drop=True)

        # Split data: 70% train, 20% validation, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)  # 70% train, 30% remaining
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)  # Split remaining into 20% val, 10% test

        # Transforms
        transform = CustomTransform(resize=(256, 256), normalize=True)

        # Datasets
        train_dataset = RotatedImageDataset(train_df, img_dir, transform=transform)
        val_dataset = RotatedImageDataset(val_df, img_dir, transform=transform)
        test_dataset = RotatedImageDataset(test_df, img_dir, transform=transform)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    # ───────────────────────────────
    # Train Model (with Early Stopping)
    # ───────────────────────────────
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cpu", patience=10, test_loader=None):
        model.to(device)
        train_losses, val_losses = [], []

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in tqdm(range(num_epochs), total=num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            for images, angles, _ in train_loader:
                images, angles = images.to(device), angles.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, angles)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, angles, _ in val_loader:
                    images, angles = images.to(device), angles.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, angles)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    #print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # ───────────────
        # Test Evaluation
        # ───────────────
        predictions = []
        avg_test_loss = None
        if test_loader:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, angles, filenames in test_loader:
                    images, angles = images.to(device), angles.to(device)
                    outputs = model(images)

                    # Loss
                    loss = criterion(outputs, angles)
                    test_loss += loss.item()

                    # Convert predictions back to degrees
                    outputs = outputs.cpu().numpy()
                    angles = angles.cpu().numpy()

                    for out, tgt, fname in zip(outputs, angles, filenames):
                        pred_angle_rad = np.arctan2(out[0], out[1])
                        pred_angle_deg = np.rad2deg(pred_angle_rad) % 360

                        actual_angle_rad = np.arctan2(tgt[0], tgt[1])
                        actual_angle_deg = np.rad2deg(actual_angle_rad) % 360

                        predictions.append({
                            "filename": fname,
                            "predicted_angle": float(pred_angle_deg),
                            "actual_angle": float(actual_angle_deg),
                            "error": float(pred_angle_deg - actual_angle_deg)
                        })

            avg_test_loss = test_loss / len(test_loader)

        return train_losses, val_losses, avg_test_loss, predictions

    # Step 5: Inference (Prediction)
    def predict_angle(model, image_path, transform, device="cpu"):
        model.eval()

        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image).cpu().numpy()[0]   # [sinθ, cosθ]
            pred_angle_rad = np.arctan2(output[0], output[1])
            pred_angle_deg = np.rad2deg(pred_angle_rad) % 360
        return pred_angle_deg


    # Step 6: Main Script
    # if __name__ == "__main__":
    # Paths
    csv_file = os.path.join(train_temp_dir, "rotated_dataset_labels.csv")  # CSV file with filenames and angles
    img_dir = train_temp_dir  # Directory containing the images

    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_data_loaders(csv_file, img_dir, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = DocumentAngleCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Train the model and collect loss history
    train_losses, val_losses, test_loss, predictions = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        test_loader=test_loader
    )

    # Save the model
    # we create the temp_dir for our output because the torch.save expects local directories and no foundry datasets 
    output_dir = tempfile.mkdtemp()

    model_path = os.path.join(output_dir, "rotation_model.pth")
    torch.save(model.state_dict(), model_path)

    # save plot epoch vs loss
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Loss")
    plt.legend()
    plt.grid(True)
    loss_plot = os.path.join(output_dir, "epoch_vs_loss.png")
    plt.savefig(loss_plot)

    # save the avg test loss
    with open(output_dir + "/" + "avg_test_loss.txt", "w") as t_loss:
        t_loss.write(str(test_loss))

    # save the test_predictions vs actual angles
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # ────────────────────────────────
    # Save checkpoint and logs to checkpoint dataset.
    # ────────────────────────────────
    output_filepaths = read_dir_recursive(output_dir)

    # Save the new checkpoints
    write_files_to_dataset(
    dest_dataset=rotation_model,
    files_to_write=output_filepaths,
    )
