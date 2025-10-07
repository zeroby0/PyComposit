import torch
from . import matmul

__all__ = ["QLinear", "QConv2d", "QConv1d"]

class QLinear(torch.nn.Module):
    def __init__(self, linear_layer, scale):
        super().__init__()

        self.weight = torch.nn.Parameter(linear_layer.weight.clone())
        self.bias = torch.nn.Parameter(linear_layer.bias.clone()) if linear_layer.bias is not None else None

        self.scale = scale
        self.is_highprecision = True

    @torch.compile
    def forward(self, x):
        # print('QLIN')

        matmul_fn = matmul.MATMUL if self.is_highprecision else matmul.MATMUL_LP

        return matmul_fn.apply(
                x,
                self.weight.t(),
                self.bias
                if self.bias is not None
                else torch.zeros(self.weight.size(0), device=x.device, dtype=x.dtype),
                self.scale,
        )

    def __repr__(self):
        return f"QLinear(in_features=?, out_features=?, bias={self.bias is not None}) @composit scale={self.scale:0.4f} highprecision={self.is_highprecision}"


class QConv2d(torch.nn.Module):
    def __init__(self, conv_layer, scale):
        super().__init__()

        self.weight = torch.nn.Parameter(conv_layer.weight.clone())
        self.bias = torch.nn.Parameter(conv_layer.bias.clone()) if conv_layer.bias is not None else None
        self.scale = scale
        self.is_highprecision = True

        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.padding_mode = conv_layer.padding_mode
        self.kernel_size = conv_layer.kernel_size
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels

    @torch.compile
    def forward(self, x):
        N, C, H, W = x.shape

        if self.padding_mode != "zeros":
            x = torch.nn.functional.pad(
                x,
                (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                mode=self.padding_mode,
            )
            padding = 0
        else:
            padding = self.padding

        kh, kw = self.kernel_size
        dh, dw = self.dilation
        effective_kh = (kh - 1) * dh + 1
        effective_kw = (kw - 1) * dw + 1

        pad_h = padding[0] if isinstance(padding, tuple) else padding
        pad_w = padding[1] if isinstance(padding, tuple) else padding

        out_h = (H + 2 * pad_h - effective_kh) // self.stride[0] + 1
        out_w = (W + 2 * pad_w - effective_kw) // self.stride[1] + 1

        x_unfolded = torch.nn.functional.unfold(
            x, kernel_size=self.kernel_size, padding=padding, stride=self.stride, dilation=self.dilation
        )

        # Shape: (N, C_in * kh * kw, out_h * out_w)
        x_unfolded = x_unfolded.view(N, self.groups, C // self.groups * kh * kw, out_h * out_w)
        x_unfolded = x_unfolded.permute(0, 1, 3, 2).contiguous()
        x_unfolded = x_unfolded.view(N * self.groups * out_h * out_w, -1)

        # Shape weights: (groups, out_channels/groups, C_in/groups * kh * kw)
        weight_grouped = self.weight.view(self.groups, self.out_channels // self.groups, -1)
        weight_grouped = weight_grouped.transpose(1, 2).contiguous()
        weight_grouped = weight_grouped.view(self.groups, -1, self.out_channels // self.groups)
        weight_grouped = weight_grouped.repeat_interleave(N * out_h * out_w, dim=0)
        weight_grouped = weight_grouped.view(-1, self.out_channels // self.groups)

        bias_grouped = (
            self.bias.view(self.groups, -1)
            .repeat_interleave(N * out_h * out_w, dim=0)
            .view(-1, self.out_channels // self.groups)
            if self.bias is not None
            else torch.zeros(
                N * self.groups * out_h * out_w,
                self.out_channels // self.groups,
                device=x.device,
                dtype=x.dtype,
            )
        )

        matmul_fn = matmul.MATMUL_BATCH if self.is_highprecision else matmul.MATMUL_LP

        out = matmul_fn.apply(x_unfolded, weight_grouped, bias_grouped, self.scale)

        out = out.view(N, self.groups, out_h * out_w, -1)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(N, self.out_channels, out_h, out_w)

        return out

    def __repr__(self):
        return (
            f"QConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, padding_mode='{self.padding_mode}', "
            f"bias={self.bias is not None}) @composit scale={self.scale:0.4f} highprecision={self.is_highprecision}"
        )

class QConv1d(torch.nn.Module):
    def __init__(self, conv_layer, scale):
        super().__init__()

        if conv_layer.dilation != (1,):
            raise ValueError(f"Only dilation=1 supported, got {conv_layer.dilation}")
        if conv_layer.groups != 1:
            raise ValueError(f"Only groups=1 supported, got {conv_layer.groups}")
        if conv_layer.padding_mode != "zeros":
            raise ValueError(f"Only padding_mode='zeros' supported, got {conv_layer.padding_mode}")

        self.weight = torch.nn.Parameter(conv_layer.weight.clone())
        self.bias = torch.nn.Parameter(conv_layer.bias.clone()) if conv_layer.bias is not None else None
        self.scale = scale
        self.is_highprecision = True

        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.kernel_size = conv_layer.kernel_size
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels

    def forward(self, x):
        N, C, L = x.shape
        out_l = (L + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1

        x_unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size[0], 1),
            padding=(self.padding[0], 0),
            stride=(self.stride[0], 1),
        ).squeeze(-2)

        x_unfolded = x_unfolded.transpose(1, 2)

        weight_flat = self.weight.view(self.out_channels, -1).t()

        matmul_fn = matmul.MATMUL_BATCH if self.is_highprecision else matmul.MATMUL_LP

        out = matmul_fn.apply(
            x_unfolded,
            weight_flat,
            self.bias
            if self.bias is not None
            else torch.zeros(self.out_channels, device=x.device, dtype=x.dtype),
            self.scale,
        )

        out = out.transpose(1, 2).view(N, self.out_channels, out_l)

        return out

    def __repr__(self):
        return (
            f"QConv1d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}) @composit scale={self.scale:0.4f} highprecision={self.is_highprecision}"
        )