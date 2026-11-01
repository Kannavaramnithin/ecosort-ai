import os
import zipfile

SOURCE = r"F:\garbage_classifier_model\garbage_classifier_model"
OUTPUT = r"F:\ECOSORTAI\ECO\best_model.pth"

print("Creating PyTorch model file...")

with zipfile.ZipFile(
    OUTPUT,
    "w",
    compression=zipfile.ZIP_STORED
) as z:

    for root, dirs, files in os.walk(SOURCE):

        for file in files:

            full_path = os.path.join(root, file)

            relative_path = os.path.relpath(
                full_path,
                SOURCE
            )

            # PyTorch requires files inside a subdirectory
            archive_path = os.path.join(
                "archive",
                relative_path
            )

            z.write(
                full_path,
                archive_path
            )

print("Model created:")
print(OUTPUT)