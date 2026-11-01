import torch

path = r"F:\ECOSORTAI\ECO\best_model.pth"

print("Loading model...")

try:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("SUCCESS!")
    print("Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Keys:")
        for key in checkpoint.keys():
            print(" -", key)

except Exception as e:
    print("ERROR:")
    print(e)