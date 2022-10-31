import pynvml

def get_free_gpu() -> str:
    """
    Get the GPU possessing the maximum free memory, or "cpu" if none of the GPU is available.
    Returns:
        str
    """

    try:
        pynvml.nvmlInit()

        deviceCount = pynvml.nvmlDeviceGetCount()
        max_idx = None
        max_ratio = 0
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = info.total
            free = info.free
            ratio = free / total
            if ratio > max_ratio and ratio > 0.5:
                max_ratio = ratio
                max_idx = i

        if max_idx is not None:
            return f'cuda:{max_idx}'
    except:
        return 'cpu'