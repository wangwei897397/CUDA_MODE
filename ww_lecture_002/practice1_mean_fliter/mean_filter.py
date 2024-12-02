from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.cpp_extension import load_inline
import os


def compile_extension():
    cuda_source =  Path("/home/wei.wang39/code/lectures/lecture_002/mean_filter/mean_filter_kernel.cu").read_text()
    cpp_source = "torch::Tensor mean_filter(torch::Tensor image, int radius);"
    build_path = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(f'{build_path}/cuda_build', exist_ok=True)
    mean_filter_extension = load_inline(
        name="mean_filter_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["mean_filter"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory=f'{build_path}/cuda_build',
    )
    return mean_filter_extension

if __name__ == "__main__":
    # mean filter kernel
    ext = compile_extension()

    image = Image.open("ww_lecture_002/Grace_Hopper.jpg")
    inputs = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous().cuda()

    assert inputs.dtype == torch.uint8
    print("Input image:", inputs.shape, inputs.dtype)

    result = ext.mean_filter(inputs, 8)
    print("Output image:", result.shape, result.dtype)

    result = result.cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(result)
    image.save("ww_lecture_002/practice1_mean_fliter/output.png")



