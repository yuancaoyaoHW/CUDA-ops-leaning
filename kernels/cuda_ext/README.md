# PyTorch C++/CUDA Extension Lab

Use this directory for small PyTorch extension experiments after the Triton baseline is verified.

Recommended order:

1. Build a CPU-only C++ extension.
2. Add one minimal CUDA kernel.
3. Compare every result against a PyTorch reference operation.
4. Keep each extension small enough to inspect end to end.
