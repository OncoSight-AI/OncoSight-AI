from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


class UNetSegmentationService:
    def __init__(self, model_path, threshold=0.50, image_size=256):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.is_file():
            raise FileNotFoundError(f"U-Net checkpoint not found: {self.model_path}")

        self.model = UNet(in_channels=1, out_channels=1).to(self.device)

        checkpoint = torch.load(
            str(self.model_path),
            map_location=self.device,
            weights_only=False
        )

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"U-Net loaded: {self.model_path}")
        print(f"Inference device: {self.device}")

        def predict_batch(self, slices):
        if len(slices) == 0:
            return []

        predictions = []

        with torch.inference_mode():
            for item in slices:
                height, width = item.shape

                batch = np.asarray(
                    item, dtype=np.float32
                )[None, None, :, :]

                tensor = torch.from_numpy(batch)

                tensor = F.interpolate(
                    tensor,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False
                )

                logits = self.model(tensor)
                probabilities = torch.sigmoid(logits)

                restored = F.interpolate(
                    probabilities,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False
                )

                mask = (
                    restored[0, 0] >= self.threshold
                ).to(torch.uint8).numpy()

                predictions.append(mask)

                del tensor, logits, probabilities, restored

        return predictions