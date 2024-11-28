import triton
import triton.language as tl
import torch

@triton.jit
def triton_square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)

def triton_square(matrix):
    height, width = matrix.shape

    # set block size
    BLOCK_SIZE = triton.next_power_of_2(width)
    # set number of warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    result = torch.empty_like(matrix)

    triton_square_kernel[(height, )](
        result,
        matrix,
        matrix.stride(0),
        result.stride(0),
        width,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return result

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=[
            'triton',
            'torch-native',
            'torch-compile',
        ],
        line_names=[
            'Triton',
            'Torch (native)',
            'Torch (compiled)',
        ],
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],
        ylabel="GB/s",
        plot_name="square() performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.square(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_square(x), quantiles=quantiles)
    if provider == 'torch-compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.compile(torch.square)(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = triton_square(x)
    y_torch = torch.square(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    triton_profile_save_path = "/home/wei.wang39/code/lectures/ww_lecture_001"
    benchmark.run(show_plots=True, print_data=True, save_path=triton_profile_save_path)