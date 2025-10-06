import torch
from icecream import ic

torch.set_printoptions(profile="full", sci_mode=False)
torch.set_float32_matmul_precision("high")

__all__ = ["round_to_quant", "quantize", "dequantize", "snap_to_component"]

posit_41, _ = torch.sort(torch.tensor(
    [-16.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.0625, 0.0, 0.0625, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0],
    dtype=torch.float32,
    device="cuda",
))



def get_qgrid(posit_4x, scale):
    grid = posit_4x.unsqueeze(1) + scale * posit_4x.unsqueeze(0)
    grid_flat = grid.flatten()
    grid_sorted, sort_idx = torch.sort(grid_flat)

    return grid_sorted, sort_idx


def snap_to_nearest(tensor, target):
    tensor_flat = tensor.flatten()
    indices = torch.searchsorted(target, tensor_flat)

    indices = torch.clamp(indices, 0, len(target) - 1)
    indices_prev = torch.clamp(indices - 1, 0, len(target) - 1)

    dist_current = torch.abs(tensor_flat - target[indices])
    dist_prev = torch.abs(tensor_flat - target[indices_prev])

    closest_indices = torch.where(dist_prev < dist_current, indices_prev, indices)

    return target[closest_indices].reshape(tensor.shape), closest_indices


def snap_to_component(tensor, posit_4x=None):
    posit_4x_temp = posit_41 if posit_4x is None else posit_4x

    tensor.clamp(posit_4x_temp.min(), posit_4x_temp.max())
    tensor.clamp(posit_4x_temp.min(), posit_4x_temp.max())

    result, _ = snap_to_nearest(tensor, posit_4x)

    return result


def snap_to_quant(tensor, posit_4x, scale):
    qgrid, _ = get_qgrid(posit_4x, scale)

    result, _ = snap_to_nearest(tensor, qgrid)

    return result


def snap_to_quant_split(tensor, posit_4x, scale):
    qgrid, qgrid_idx = get_qgrid(posit_4x, scale)

    _, indices = snap_to_nearest(tensor, qgrid)

    qgrid_idx_major = qgrid_idx // 15
    qgrid_idx_minor = qgrid_idx % 15

    tensor_major = posit_4x[qgrid_idx_major[indices]].reshape(tensor.shape)
    tensor_minor = posit_4x[qgrid_idx_minor[indices]].reshape(tensor.shape)

    return tensor_major, tensor_minor


@torch.compile
def round_to_quant(tensor, posit_4x=None, scale=1 / 24.1):
    return snap_to_quant(tensor, posit_41 if posit_4x is None else posit_4x, scale)


@torch.compile
def quantize(tensor, posit_4x=None, scale=1 / 24.1):
    return snap_to_quant_split(tensor, posit_41 if posit_4x is None else posit_4x, scale)


@torch.compile
def dequantize(tensor_major, tensor_minor, posit_4x=None, scale=1 / 24.1):

    posit_4x_temp =  posit_41 if posit_4x is None else posit_4x

    # ic(posit_4x, posit_4x_temp)

    tensor_major.clamp(posit_4x_temp.min(), posit_4x_temp.max())
    tensor_minor.clamp(posit_4x_temp.min(), posit_4x_temp.max())

    return snap_to_component(tensor_major, posit_4x_temp) + scale * snap_to_component(tensor_minor, posit_4x_temp)


if __name__ == "__main__":
    import time
    from icecream import ic

    # Example usage:
    huge_tensor = torch.randn(128, 128, 128, device="cuda")

    start = time.monotonic()
    for i in range(10000):
        result = snap_to_quant(huge_tensor, posit_41, 1/24.1)
    stop = time.monotonic()
    print(stop - start)

    start = time.monotonic()
    for i in range(10000):
        result = snap_to_quant_split(huge_tensor, posit_41, 1/24.1)
    stop = time.monotonic()
    print(stop - start)

    # Curiously, which function is faster depends on the
    # number of calls. For 10k, split is faster. Possibly
    # because it's caching posit_4x.
    # For 100k, the other function is a bit faster, it's
    # probably only caching the larger qgrid when it's
    # called a lot?

    ic(get_qgrid(posit_41, 1/24.1))
