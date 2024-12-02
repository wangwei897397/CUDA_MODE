from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("/home/wei.wang39/code/lectures/lecture_002/mean_filter/mean_filter_kernel.cu").read_text()
    cpp_source = "torch::Tensor mean_filter(torch::Tensor image, int radius);"

    # Load the CUDA kernel as a PyTorch extension
    rgb_to_grayscale_extension = load_inline(
        name="mean_filter_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["mean_filter"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rgb_to_grayscale_extension


def main():
    """
    Use torch cpp inline extension function to compile the kernel in mean_filter_kernel.cu.
    Read input image, convert apply mean filter custom cuda kernel and write result out into output.png.
    """
    ext = compile_extension()

    # x = read_image("Grace_Hopper.jpg").contiguous().cuda()
    image = Image.open("lecture_002/mean_filter/Grace_Hopper.jpg")
    x = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous().cuda()

    assert x.dtype == torch.uint8
    print("Input image:", x.shape, x.dtype)

    y = ext.mean_filter(x, 8)

    print("Output image:", y.shape, y.dtype)
    y = y.cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(y)
    image.save("lecture_002/mean_filter/output.png")


if __name__ == "__main__":
    main()
