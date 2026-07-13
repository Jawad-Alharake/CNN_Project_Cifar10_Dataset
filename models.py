import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    """Custom Convolutional Neural Network built from scratch."""
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: Input 3x32x32 -> output channel: 32x32x32 -> Pool: 32x16x16
            # CIFAR-10 images start as 3 color channels at 32x32 pixels.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32x16x16 -> output channel: 64x16x16 -> Pool: 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64x8x8 -> output channel: 128x8x8 -> Pool: 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier Network (Dual-Layer Pyramidal Architecture)
        self.classifier = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(128, 512), # <-- Changed from 2048 to 128 to match GAP plumbing
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45), # Strong regularization on the high-capacity layer
            
            # Fully Connected Layer 2
            nn.Linear(512, 256), # standard procedure is to decrease the number of neurons by half
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25), # Moderate regularization on the intermediate layer
            
            # Final Output Layer
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # 1. Shrink the spatial dimensions to 1x1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) # Shrinks to 128x1x1
        # 2. Squeeze the empty dimensions out to make it a flat 1D vector of 128
        x = torch.flatten(x, 1) # Used global average pooling since the data is small and used only two dense layer (flatten uses the whole number of matrices, while global average pooling gives me the understanding of the image without the position so we lose some information, but if we don't care where the position is, then it's ok.)
        # 3. Pass to classifier
        x = self.classifier(x)
        return x


def get_transfer_learning_model(num_classes=10, feature_extract=True):
    """
    Loads pre-trained MobileNetV2 model and replaces the final classifier layer.
    feature_extract = True: Freezes backbone layers, trains only the final classifier.
    feature_extract = False: Fine-tunes the entire model.
    """
    # Load pre-trained weights for MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the upstream features if feature extracting
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
    # MobileNetV2's classifier is a Sequential block: (0): Dropout, (1): Linear
    # We replace the Linear layer to match CIFAR-10's 10 classes
    # First, How many input wires are coming into you from the frozen network? The answer is 1,280. We save that number as num_ftrs.
    # Second, we completely overwrite the variable at model.classifier[1]. The old 1,000-class layer is permanently deleted from memory.
    # We replace it with a brand new nn.Linear layer that accepts the 1,280 incoming wires and outputs exactly num_classes (which is 10).
    # 1. Measure the plumbing connections
    num_ftrs = model.classifier[1].in_features
    # 2. Swap the part
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model


if __name__ == '__main__':
    print("==================================================")
    print("      Testing Model Architectures Pipeline        ")
    print("==================================================\n")

    # ---- 1. Test the Custom CNN ----
    print("1. Initializing Custom CNN...")
    custom_net = CustomCNN(num_classes=10)
    print(custom_net) # This prints the entire layer-by-layer layout
    
    # Updated to 128x128 to match the new dataset pipeline
    fake_image = torch.randn(1, 3, 128, 128)
    custom_output = custom_net(fake_image)
    print(f"Input Shape: {fake_image.shape}")
    print(f"Output Shape (Should be [1, 10]): {custom_output.shape}\n")
    print("-" * 50)

    # ---- 2. Test the MobileNetV2 Transfer Learning Model ----
    print("2. Initializing MobileNetV2 Transfer Learning Model...")
    mobilenet_tl = get_transfer_learning_model(num_classes=10, feature_extract=True)
    
    # Updated to match the new 128x128 input resolution pipeline
    fake_scaled_image = torch.randn(1, 3, 128, 128)
    tl_output = mobilenet_tl(fake_scaled_image)
    print(f"Input Shape: {fake_scaled_image.shape}")
    print(f"Output Shape (Should be [1, 10]): {tl_output.shape}\n")
    
    print("Both models compiled and verified successfully without shape mismatches!")