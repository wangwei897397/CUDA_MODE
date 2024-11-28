import torch

def time_torch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    return elapsed_time

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

def square_4(a):
    return torch.pow(a, 2)

def square_5(a):
    return torch.square(a)

def profile_function(func, data):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        for i in range(10):
            func(data)
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    # step1 
    data1 = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
    print(torch.square(data1)) # tensor([1., 4., 9.])
    print(data1 ** 2) # tensor([1., 4., 9.])
    print(data1 * data1) # tensor([1., 4., 9.])
    print(torch.pow(data1, 2)) # tensor([1., 4., 9.])

    # step2
    data2 = torch.randn(10000, 10000).cuda()
    print("a * a: ", time_torch_function(square_2, data2))
    print("a ** 2: ", time_torch_function(square_3, data2))
    print("torch.pow: ", time_torch_function(square_4, data2))
    print("torch.square: ", time_torch_function(square_5, data2))

    # step3
    print("=============")
    print("Profiling a * a")
    print("=============")
    profile_function(square_2, data2)

    print("=============")
    print("Profiling a ** 2")
    print("=============")
    profile_function(square_3, data2)

    print("=============")
    print("Profiling torch.pow")
    print("=============")
    profile_function(square_4, data2)

    print("=============")
    print("Profiling torch.square")
    print("=============")
    profile_function(square_5, data2)

