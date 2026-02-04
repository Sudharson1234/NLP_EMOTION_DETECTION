import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[FILE] {name} - {obj.shape}")
    else:
        print(f"[DIR] {name}")

print("Inspecting emotion_model_pretrained.h5...")
with h5py.File("emotion_model_pretrained.h5", "r") as f:
    f.visititems(print_structure)
