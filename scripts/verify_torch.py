import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("matmul on:", y.device, "sum:", float(y.sum()))
    print("VRAM allocated (MB):", round(torch.cuda.memory_allocated() / 1e6, 1))
else:
    print("CPU only")
