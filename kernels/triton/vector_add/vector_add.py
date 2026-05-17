import torch
import triton
import triton.language as tl


@triton.jit()
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_vector_add(x: torch.Tensor, y: torch.Tensor):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.device.type != "cuda":
        raise ValueError("x and y must be CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    torch.manual_seed(0)
    size = 98432
    device = torch.device("cuda")
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output_torch = x + y
    output_triton = triton_vector_add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )


if __name__ == "__main__":
    main()
