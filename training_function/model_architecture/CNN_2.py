import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

features_columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]


class CNNModel(nn.Module):
    def __init__(
        self,
        in_channels=6,
        kernel_size=[3, 3, 3, 3],
        num_classes=2,
        model_param_list=[64, 128, 256, 128, 64, 32],
    ):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=model_param_list[0],
                kernel_size=kernel_size[0],
                stride=1,
            ),
            nn.BatchNorm1d(model_param_list[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=model_param_list[0],
                out_channels=model_param_list[1],
                kernel_size=kernel_size[1],
                stride=1,
            ),
            nn.BatchNorm1d(model_param_list[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=model_param_list[1],
                out_channels=model_param_list[2],
                kernel_size=kernel_size[2],
                stride=1,
            ),
            nn.BatchNorm1d(model_param_list[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=model_param_list[2],
                out_channels=model_param_list[3],
                kernel_size=kernel_size[3],
                stride=1,
            ),
            nn.BatchNorm1d(model_param_list[3]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.swing_mode_embedding = nn.Embedding(num_embeddings=10, embedding_dim=16)

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=model_param_list[3] + 16, out_features=model_param_list[4]
            ),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(
                in_features=model_param_list[4], out_features=model_param_list[5]
            ),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(in_features=model_param_list[5], out_features=num_classes),
        )

    def forward(self, x, swing_mode):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        mode_embed = self.swing_mode_embedding(swing_mode)
        combined = torch.cat([x, mode_embed], dim=1)
        out = self.fc(combined)
        return out


class CNNDataset(Dataset):
    def __init__(self, sensor_dataframe, mode_dataframe, labels_dataframe):
        self.sensor_data = []
        for idx in range(len(sensor_dataframe)):
            row = sensor_dataframe.iloc[idx]
            lengths = [len(row[col]) for col in features_columns]
            assert len(set(lengths)) == 1, f"样本 {idx} 传感器数据长度不一致"
            sensor_array = np.stack(
                [row[col] for col in features_columns], axis=0
            ).astype(np.float32)
            self.sensor_data.append(sensor_array)
        self.mode = mode_dataframe.to_numpy().flatten().astype(np.int64)
        self.labels = labels_dataframe.to_numpy().flatten().astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sensor": self.sensor_data[idx],
            "mode": self.mode[idx],
            "labels": self.labels[idx],
        }


def collate_fn(batch):
    sensors = [item["sensor"] for item in batch]
    modes = [item["mode"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_length = max(sensor.shape[1] for sensor in sensors)

    padded_sensors = []
    for sensor in sensors:
        pad = max_length - sensor.shape[1]
        if pad > 0:
            padded = np.pad(
                sensor, ((0, 0), (0, pad)), mode="constant", constant_values=0
            )
        else:
            padded = sensor
        padded_sensors.append(padded)

    return {
        "sensor": torch.tensor(np.stack(padded_sensors), dtype=torch.float32),
        "mode": torch.tensor(modes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_cnn_model(
    x,
    y,
    kernel_size=[3, 3, 3, 3],
    num_classes=2,
    model_param_list=[32, 64, 128, 256, 128, 64],
    batch_size=50,
    num_epochs=20,
    model_path="best_model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用 {device} 進行訓練")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    model = CNNModel(
        kernel_size=kernel_size,
        num_classes=num_classes,
        model_param_list=model_param_list,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    features_columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    train_dataset = CNNDataset(x_train[features_columns], x_train["mode"], y_train)
    val_dataset = CNNDataset(x_test[features_columns], x_test["mode"], y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    best_val_loss = float("inf")

    early_stop_patience = 5
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["sensor"].to(device)
            swing_modes = batch["mode"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(features, swing_modes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["sensor"].to(device)
                swing_modes = batch["mode"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(features, swing_modes)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break

    print("Training complete")
    return model, kernel_size, model_param_list


def test_cnn_model(
    x,
    batch_size=1,
    kernel_size=[3, 3, 3, 3],
    num_classes=2,
    model_param_list=[32, 64, 128, 256, 128, 64],
    features_name="",
    model_path="best_model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用 {device} 進行預測")
    features_columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    predict_dataset = CNNDataset(
        x[features_columns], x["mode"], pd.Series([0] * len(x["mode"]))
    )
    predict_loader = DataLoader(
        predict_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    model = CNNModel(
        kernel_size=kernel_size,
        num_classes=num_classes,
        model_param_list=model_param_list,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    all_probabilities = []
    with torch.no_grad():
        for batch in predict_loader:
            features = batch["sensor"].to(device)
            swing_modes = batch["mode"].to(device)
            outputs = model(features, swing_modes)

            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    probabilities_array = np.concatenate(all_probabilities, axis=0)

    prob_df = pd.DataFrame(
        probabilities_array,
        columns=[f"{features_name}_{i}" for i in range(num_classes)],
    )
    return prob_df
