#include<c10/cuda/CUDAException.h>
#include<c10/cuda/CUDAStream.h>


__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height){
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height){
        int outputOffset = row * width + col;
        int inputOffset = (row * width + col) * channels;

        unsigned char r = input[inputOffset + 0];
        unsigned char g = input[inputOffset + 1];
        unsigned char b = input[inputOffset + 2];

        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);

    }
    /* 
    二维图像的线性索引：

        对于一个二维图像，像素的位置通常由行索引 (row) 和列索引 (col) 来表示。
        为了在一维数组中表示二维图像，我们需要将二维索引转换为一维索引。
        计算公式为 row * width + col，其中 width 是图像的宽度。这个公式将二维索引 (row, col) 转换为一维索引。
        多通道图像：

        彩色图像通常有多个通道，例如 RGB 图像有三个通道（红色、绿色和蓝色）。
        每个像素在内存中占据多个连续的位置，每个通道对应一个位置。
        channels 表示图像的通道数，对于 RGB 图像，channels 的值为 3。
        计算 inputOffset：

        inputOffset 表示当前像素在输入图像一维数组中的起始位置。
        计算公式 (row * width + col) * channels 包含两个部分：
        row * width + col：计算当前像素在二维图像中的线性索引。
        * channels：将线性索引乘以通道数，得到当前像素在一维数组中的起始位置。
    */
}


inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_grayscale(torch::Tensor image){
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}