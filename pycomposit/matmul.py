import torch
import snoop
from icecream import ic
from . import qt

torch.set_printoptions(profile='full', sci_mode=False)
torch.set_float32_matmul_precision("high")

__all__ = ["MATMUL", "MATMUL_LP"]


# This kernel performs A@B + C in quantised domain
# and returns the answer back as a normal unquantised float
@torch.compile
def matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, scale=1/24.1):
    A_major, A_minor = qt.quantize(A, scale=scale)
    B_major, B_minor = qt.quantize(B, scale=scale)
    C_major, C_minor = qt.quantize(C, scale=scale)

    R_major = (A_major @ B_major + C_major)
    R_minor = (A_minor @ B_major) + (A_major @ B_minor + C_minor)
    
    # ic(A_major, A_minor)
    # ic(B_major, B_minor)
    # ic(C_major, C_minor)
    # ic(R_major, R_minor)

    return qt.dequantize(R_major, R_minor, scale=scale)


# @torch.compile
# def matmul_lp(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, scale=1/24.1):
#     A_major, _ = qt.quantize(A, scale=scale)
#     B_major, _ = qt.quantize(B, scale=scale)
#     C_major, _ = qt.quantize(C, scale=scale)

#     R_major = (A_major @ B_major + C_major)

#     return qt.dequantize(R_major, torch.zeros_like(R_major), scale=scale)


class MATMUL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, scale):
        ctx.save_for_backward(A, B, C)

        # Float 32
        # return A.detach() @ B.detach() + C.detach()

        # Float 16
        # return (A.half() @ B.half() + C.half()).to(torch.float32).detach()
        
        # Round to Composit
        # return (qt.round_to_quant(A, scale=scale) @ qt.round_to_quant(B, scale=scale) + qt.round_to_quant(C, scale=scale)).detach()

        # Composit MAC
        return matmul(A.detach(), B.detach(), C.detach(), scale)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, C = ctx.saved_tensors

        # ic(A.shape, B.shape, C.shape, grad_output.shape)
        # ic(B.transpose(-2, -1).shape)
        # ic(A.transpose(-2, -1).shape)
        # ic(ctx.needs_input_grad)
        # ic('-------------------- MM')

        grad_A = grad_output @ B.transpose(-2, -1) if ctx.needs_input_grad[0] else None
        grad_B = A.transpose(-2, -1) @ grad_output if ctx.needs_input_grad[1] else None
        grad_C = grad_output if ctx.needs_input_grad[2] else None

        return grad_A, grad_B, grad_C, None


class MATMUL_BATCH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, scale):
        ctx.save_for_backward(A, B, C)

        # Float 32
        # return A.detach() @ B.detach() + C.detach()

        # Float 16
        # return (A.half() @ B.half() + C.half()).to(torch.float32).detach()
        
        # Round to Composit
        # return (qt.round_to_quant(A, scale=scale) @ qt.round_to_quant(B, scale=scale) + qt.round_to_quant(C, scale=scale)).detach()

        # Composit MAC
        return matmul(A.detach(), B.detach(), C.detach(), scale)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, C = ctx.saved_tensors

        # ic(A.shape, B.shape, C.shape, grad_output.shape)
        # ic(B.transpose(-2, -1).shape)
        # ic(A.transpose(-2, -1).shape)
        # ic(ctx.needs_input_grad)
        # ic("-------------------- MMB")

        grad_A = grad_output @ B.transpose(-2, -1) if ctx.needs_input_grad[0] else None
        grad_B = (A.transpose(-2, -1) @ grad_output).sum(0) if ctx.needs_input_grad[1] else None
        grad_C = grad_output.sum(0) if ctx.needs_input_grad[2] else None

        return grad_A, grad_B, grad_C, None


# class MATMUL(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, B, C, scale):
#         ctx.save_for_backward(A, B, C)
#         return matmul(A.detach(), B.detach(), C.detach(), scale)

#     @staticmethod
#     def backward(ctx, grad_output):
#         A, B, C = ctx.saved_tensors

#         print(A.shape, B.shape, C.shape)

#         return torch.ones_like(A), torch.ones_like(B), torch.ones_like(C), None


#         grad_A = grad_output @ B.t() if ctx.needs_input_grad[0] else None
#         grad_B = A.t() @ grad_output if ctx.needs_input_grad[1] else None
#         grad_C = grad_output if ctx.needs_input_grad[2] else None

#         return grad_A, grad_B, grad_C, None


class MATMUL_LP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, scale):
        raise NotImplementedError
        ctx.save_for_backward(A, B)
        return matmul_lp(A.detach(), B.detach(), C.detach(), scale)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors

        grad_A = grad_output @ B.t() if ctx.needs_input_grad[0] else None
        grad_B = A.t() @ grad_output if ctx.needs_input_grad[1] else None
        grad_C = grad_output if ctx.needs_input_grad[2] else None

        return grad_A, grad_B, grad_C, None



# if __name__ == '__main__':

#     # A = torch.randn(4, device="cuda")
#     # B = torch.randn(4, device="cuda")
#     # C = torch.zeros(4, device="cuda")

#     A = torch.tensor([[-0.1608, -0.0680], [1.3257, 0.9172]], device="cuda:0")
#     B = torch.tensor([[0.1696, -3.6974], [2.0586, 0.4939]], device="cuda:0")
#     C = torch.zeros(2, 2, device="cuda")

#     # ic(A)
#     # ic(B)
#     # ic(C)

#     ic(matmul(A, B, C, scale=100 / 6401))
#     ic(matmul(A, B, C, scale=100 / 3204))
#     ic(matmul_lp(A, B, C))
#     ic(A @ B)

#     # ic(A.type(torch.float16) @ B.type(torch.float16) + C.type(torch.float16))


#     # A = tensor([[[ 0.5692, -1.7282],
#     #                 [ 1.1981, -0.4977]]], device='cuda:0')

# exit()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    for i in range(25):

        A = torch.randn(128, 128, 128, device="cuda")# * 3
        B = torch.randn(128, 128, 128, device="cuda")# * 3
        C = torch.randn(128, 128, 128, device="cuda")# * 3

        print(i)
        # print(A.max(), A.min())

        scales = []
        diffs = []
        rdiffs = []
        for iscale in range(2500, 3500):
            # if iscale == 3200: continue

            reference = A@B + C
            result = matmul(A, B, C, scale=100/iscale)

            # Relative error
            rdiff = torch.sum(torch.abs(result - reference) / torch.clamp(torch.abs(reference), min=1e-6)) / (128*128*128*128)

            diff = torch.sum(torch.abs(result - reference)) / (128*128*128*128)

            # print(iscale, diff)
            # if iscale > 3000 and iscale < 3500:
            #     scales.append(iscale)
            #     diffs.append(diff.cpu())

            # rdiff = torch.clamp(rdiff, max=2e10)

            scales.append(iscale)
            diffs.append(diff.cpu())
            rdiffs.append(rdiff.cpu())

            # if iscale > 3000 and iscale < 3500:
            #     print(iscale, diff)

        plt.plot(scales, diffs, alpha=0.6, linestyle="solid", linewidth=0.5, color="turquoise")
        # plt.plot(scales, rdiffs, alpha=0.4, linestyle="solid", linewidth=0.5, color="mediumpurple")

        # break

    custom_lines = [
        Line2D([0], [0], color="turquoise", lw=4),
        Line2D([0], [0], color="mediumpurple", lw=4),
    ]

    plt.xlabel('iscale')
    plt.ylabel('diff')
    plt.legend(custom_lines, ['Abs Diff', 'Rel Diff'])
    plt.tight_layout()
    plt.savefig('scale.png', dpi=900)


