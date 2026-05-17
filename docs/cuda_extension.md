# PyTorch C++/CUDA Extension Notes

Use this document for notes while learning PyTorch C++/CUDA extensions.

Suggested order:

1. Build a CPU-only C++ extension.
2. Add a minimal CUDA kernel.
3. Compare extension output with a PyTorch reference operation.
4. Add pytest coverage before optimizing.

Do not assume a system CUDA toolkit is installed. Check the local WSL2 environment first.
