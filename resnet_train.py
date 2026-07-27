import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Use the non-interactive backend

# --- RESNET HELPER BLOCKS ---
class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# --- RESNET MODEL CLASS ---
class ResNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        # Stem block (Adapted for 1-channel FashionMNIST input)
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual stages
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        # Head block
        self.net = nn.Sequential(
            self.b1, self.b2, self.b3, self.b4, self.b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        return self.net(X)

# --- TRAINING PIPELINE ---
def train_resnet():
    # Sanity Check: Print GPU info to the log
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Training on CPU!")

    img_size = 96
    lr = 0.05  # Standard ResNet learning rate initialization

    model = ResNet(lr=lr)

    # Force execution safety if system visibility lacks GPUs
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = d2l.Trainer(max_epochs=10, num_gpus=num_gpus)

    # Disable the interactive plotting for non-Jupyter environments
    model.board.display = False 
    data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))

    trainer.fit(model, data)

    # --- SAVE FINAL PLOT AS IMAGE ---
    if hasattr(model, 'board') and model.board.data:
        plt.figure(figsize=(10, 6))

        for label, points in model.board.data.items():
            x = [p.x for p in points]
            y = [p.y for p in points]
            plt.plot(x, y, label=label)

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('ResNet Training Metrics')
        plt.legend()
        plt.grid(True)

        plt.savefig('training_plot.png')
        plt.close() 
        print("Training plot manually reconstructed and saved to training_plot.png")

    # --- SAFE SAVE METRICS TO TXT ---
    if hasattr(model, 'board'):
        with open('epoch_metrics.txt', 'w') as f:
            f.write("Training Metrics History\n")
            f.write("========================\n")

            data_dict = model.board.data
            keys = list(data_dict.keys())

            if keys:
                min_len = min(len(data_dict[k]) for k in keys)
                f.write(f"Index\t" + "\t".join(keys) + "\n")

                for i in range(min_len):
                    line = f"{i}\t"
                    values = [f"{data_dict[k][i].y:.4f}" for k in keys]
                    line += "\t".join(values)
                    f.write(line + "\n")
                print(f"Metrics saved to epoch_metrics.txt using {min_len} shared data points.")
            else:
                print("No metrics tracking data found in model.board.")

    # Save the model weights 
    torch.save(model.state_dict(), 'resnet_fashionmnist.pth')
    print("Model saved to resnet_fashionmnist.pth")

if __name__ == "__main__":
    train_resnet()
