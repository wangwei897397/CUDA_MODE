import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("/home/wei.wang39/code/lectures/ww_lecture_002/practice2_rgb_to_grayscale/rgb_to_grayscale_kernel.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    rgb_to_grayscale_extension = load_inline(
        name="rgb_to_grayscale_extension",
        cuda_sources=cuda_source,
        cpp_sources=cpp_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory="/home/wei.wang39/code/lectures/ww_lecture_002/practice2_rgb_to_grayscale/cuda_build"
    )
    return rgb_to_grayscale_extension

def main():
    # compile the extension
    ext = compile_extension()

    # read the image
    image = Image.open("ww_lecture_002/Grace_Hopper.jpg")
    inputs = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous().cuda()
    print("Input image:", inputs.shape, inputs.dtype)

    assert inputs.dtype == torch.uint8

    # apply the rgb_to_grayscale kernel
    result = ext.rgb_to_grayscale(inputs)

    print("Output image:", result.shape, result.dtype)

    result = result.cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(result)
    image.save("ww_lecture_002/practice1_mean_fliter/output.png")



if __name__ == "__main__":
    main()