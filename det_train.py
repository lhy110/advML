import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datasets import load_dataset
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dataset (COCO-style for detection)
from datasets import load_dataset
dataset = load_dataset("cppe-5")  # small object detection dataset
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def transform(examples):
    # Ensure images are RGB
    images = [img.convert("RGB") for img in examples["image"]]

    batch_annotations = []
    for i in range(len(images)):
        objs = examples["objects"][i]
        formatted_objs = []

        # Check if objects has bboxes to avoid empty list errors
        if "bbox" in objs:
            for j in range(len(objs["bbox"])):
                formatted_objs.append({
                    "bbox": objs["bbox"][j],
                    "category_id": objs["category"][j],
                    "area": objs.get("area", [0]*len(objs["bbox"]))[j],
                    "iscrowd": 0
                })

        batch_annotations.append({
            "image_id": examples["image_id"][i],
            "annotations": formatted_objs
        })

    # Process - do NOT return tensors as "pt" here if using set_transform
    # Let the processor handle the image scaling/normalization
    encoding = processor(images=images, annotations=batch_annotations, return_tensors="pt")

    # Convert tensors to lists so they can be re-batched by DataLoader
    return {k: list(v) for k, v in encoding.items()}




# Apply the transform
dataset["train"].set_transform(transform)
from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = {"pixel_values": torch.stack(pixel_values)}

    # Keep labels as a list of dictionaries
    if "labels" in batch[0]:
        encoding["labels"] = [item["labels"] for item in batch]

    return encoding

# Load in the training data, as a batch of 4, removes order bias with shuffle=True.
train_loader = DataLoader(
    dataset["train"], 
    batch_size=4, 
    shuffle=True, 
    collate_fn=collate_fn
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=91,
    ignore_mismatched_sizes=True
)

model.to(device)
# Training Loop (D2L)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
# Inference (Detection)
model.eval()

sample = dataset["test"][0]  # cppe-5 has a 'test' split with 29 images

inputs = processor(images=sample["image"], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_object_detection(
    outputs,
    target_sizes=[sample["image"].size[::-1]]
)[0]

print(results)
