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
    def __init__(self, conv_layer):
        super().__init__()

        # Copy all parameters from the original conv layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.padding_mode = conv_layer.padding_mode

        # Store weight and bias as parameters
        self.weight = torch.nn.Parameter(conv_layer.weight.clone())
        if conv_layer.bias is not None:
            self.bias = torch.nn.Parameter(conv_layer.bias.clone())
        else:
            self.register_parameter("bias", None)

        self.qname = "QConv2d"

        # Ensure padding_mode is valid
        if self.padding_mode != "zeros":
            raise NotImplementedError(f"Padding mode '{self.padding_mode}' not yet implemented for QConv2d")

    def _calculate_output_size(self, input_size):
        """Calculate output spatial dimensions"""
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if isinstance(self.stride, int):
            stride = (self.stride, self.stride)
        else:
            stride = self.stride

        if isinstance(self.padding, int):
            padding = (self.padding, self.padding)
        else:
            padding = self.padding

        if isinstance(self.dilation, int):
            dilation = (self.dilation, self.dilation)
        else:
            dilation = self.dilation

        h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        return h_out, w_out

    # @torch.compile
    def forward(self, x):
        print("QConv", x.shape)
        batch_size, in_channels, height, width = x.shape

        h_out, w_out = self._calculate_output_size((height, width))
        out_spatial_size = h_out * w_out

        # Unfold the input to columns (im2col)
        # Shape: [batch_size, kernel_size * kernel_size * in_channels, out_spatial_size]
        unfolded = torch.nn.functional.unfold(
            x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride
        )

        # Reshape weight for matmul: [out_channels, kernel_size * kernel_size * in_channels // groups]
        weight_flat = self.weight.view(self.out_channels, -1)

        if self.bias is not None:
            bias = self.bias
        else:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)

        matmul_fn = matmul.MATMUL

        if self.groups == 1:
            # Standard convolution - single matmul
            # unfolded: [batch_size, in_channels * kH * kW, out_spatial_size]
            # weight_flat: [out_channels, in_channels * kH * kW]
            # We want: result[b, c_out, spatial] = sum_{c_in, k} weight[c_out, c_in, k] * unfolded[b, c_in*k, spatial]

            # Transpose unfolded to [batch_size, out_spatial_size, in_channels * kH * kW]
            unfolded_t = unfolded.transpose(1, 2)

            # Reshape to [batch_size * out_spatial_size, in_channels * kH * kW]
            unfolded_flat = unfolded_t.reshape(-1, unfolded_t.size(-1))

            # Apply matmul: [batch_size * out_spatial_size, out_channels]
            result_flat = matmul_fn.apply(unfolded_flat, weight_flat.t(), bias)

            # Reshape back: [batch_size, out_spatial_size, out_channels] -> [batch_size, out_channels, out_spatial_size]
            result = result_flat.view(batch_size, out_spatial_size, self.out_channels).transpose(1, 2)

        else:
            # Grouped convolution - handle each group separately
            group_in_channels = self.in_channels // self.groups
            group_out_channels = self.out_channels // self.groups

            results = []
            for g in range(self.groups):
                start_in = (
                    g * group_in_channels * self.kernel_size[0] * self.kernel_size[1]
                    if isinstance(self.kernel_size, tuple)
                    else g * group_in_channels * self.kernel_size * self.kernel_size
                )
                end_in = (
                    (g + 1) * group_in_channels * self.kernel_size[0] * self.kernel_size[1]
                    if isinstance(self.kernel_size, tuple)
                    else (g + 1) * group_in_channels * self.kernel_size * self.kernel_size
                )

                start_out = g * group_out_channels
                end_out = (g + 1) * group_out_channels

                unfolded_group = unfolded[:, start_in:end_in, :]
                weight_group = weight_flat[start_out:end_out, :]
                bias_group = (
                    bias[start_out:end_out]
                    if self.bias is not None
                    else torch.zeros(group_out_channels, device=x.device, dtype=x.dtype)
                )

                unfolded_group_t = unfolded_group.transpose(1, 2)
                unfolded_group_flat = unfolded_group_t.reshape(-1, unfolded_group_t.size(-1))

                result_group_flat = matmul_fn.apply(
                    unfolded_group_flat,
                    weight_group.t(),
                    bias_group,
                )
                result_group = result_group_flat.view(
                    batch_size, out_spatial_size, group_out_channels
                ).transpose(1, 2)
                results.append(result_group)

            # Concatenate results from all groups
            result = torch.cat(results, dim=1)

        # Fold back to output shape: [batch_size, out_channels, h_out, w_out]
        output = result.view(batch_size, self.out_channels, h_out, w_out)

        return output

    def __repr__(self):
        return (
            f"QConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}) @composit qname={self.qname}"
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
    

if __name__ == '__main__':
    from torch import nn

    class SimpleModel(nn.Module):
        def __init__(self, in_channels, out_channels, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 33, kernel_size=3, padding=1, groups=3)
            self.conv2 = nn.Conv2d(33, out_channels, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        # @snoop
        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x


    def test_conv_equivalence():
        """Test QConv2d against nn.Conv2d with various parameters"""

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_cases = [
            # Basic cases
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1},
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 2, "padding": 1},
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 0},
            # Different kernel sizes
            {"in_channels": 3, "out_channels": 6, "kernel_size": 5, "stride": 1, "padding": 2},
            {"in_channels": 3, "out_channels": 6, "kernel_size": 1, "stride": 1, "padding": 0},
            # Group convolutions
            {"in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1, "groups": 2},
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "groups": 4},
            {"in_channels": 6, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1, "groups": 3},
            # Dilated convolutions
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 2},
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2},
            # Combined cases
            {
                "in_channels": 8,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "groups": 4,
                "dilation": 2,
            },
            {
                "in_channels": 4,
                "out_channels": 8,
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "groups": 2,
                "dilation": 1,
            },
            {
                "in_channels": 14,
                "out_channels": 28,
                "kernel_size": 5,
                "stride": 3,
                "padding": 2,
                "groups": 7,
                "dilation": 1,
            },
        ]

        input_sizes = [(1, 16, 16), (2, 32, 32), (4, 8, 8)]

        all_passed = True

        for i, config in enumerate(test_cases):
            print(f"\n=== Test Case {i+1}/{len(test_cases)} ===")
            print(f"Config: {config}")

            for batch_size, H, W in input_sizes:
                # Skip incompatible configurations
                if "groups" in config and config["in_channels"] % config["groups"] != 0:
                    continue
                if "groups" in config and config["out_channels"] % config["groups"] != 0:
                    continue

                # Create input tensor
                x = torch.randn(batch_size, config["in_channels"], H, W, device=device)

                # Create original and custom conv layers
                original_conv = nn.Conv2d(**config, bias=True).to(device)
                custom_conv = QConv2d(original_conv).to(device)

                # Copy parameters to ensure they're the same
                # custom_conv.weight.data = original_conv.weight.data.clone()
                # if original_conv.bias is not None:
                #     custom_conv.bias.data = original_conv.bias.data.clone()

                # Forward pass
                with torch.no_grad():
                    original_output = original_conv(x)
                    custom_output = custom_conv(x)

                # Check shapes
                shape_match = original_output.shape == custom_output.shape
                if not shape_match:
                    print(f"  ❌ Shape mismatch: Original {original_output.shape}, Custom {custom_output.shape}")
                    all_passed = False
                    continue

                # Check values
                max_diff = torch.max(torch.abs(original_output - custom_output)).item()
                mean_diff = torch.mean(torch.abs(original_output - custom_output)).item()

                # Use relative error for better comparison
                relative_error = torch.norm(original_output - custom_output) / (
                    torch.norm(original_output) + 1e-8
                )

                tolerance = 1e-5
                if max_diff < tolerance and mean_diff < tolerance:
                    print(
                        f"  ✅ Passed - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative: {relative_error:.2e}"
                    )
                else:
                    print(
                        f"  ❌ Failed - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative: {relative_error:.2e}"
                    )
                    all_passed = False

        print(f"\n{'='*50}")
        if all_passed:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed!")

        return all_passed


    def test_edge_cases():
        """Test edge cases and potential failure modes"""

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        edge_cases = [
            # Large dilation
            {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 4, "dilation": 4},
            # No bias
            {"in_channels": 3, "out_channels": 6, "kernel_size": 3, "stride": 1, "padding": 1, "bias": False},
            # Large groups
            {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "groups": 16},
            # Asymmetric parameters
            {"in_channels": 3, "out_channels": 6, "kernel_size": (3, 5), "stride": (1, 2), "padding": (1, 2)},
        ]

        print("\n=== Testing Edge Cases ===")

        for i, config in enumerate(edge_cases):
            print(f"\nEdge Case {i+1}: {config}")

            try:
                x = torch.randn(2, config["in_channels"], 16, 16, device=device)

                # Handle bias parameter specially
                bias = config.pop("bias", True)
                original_conv = nn.Conv2d(**config, bias=bias).to(device)
                custom_conv = QConv2d(original_conv).to(device)

                # # Copy parameters
                # custom_conv.weight.data = original_conv.weight.data.clone()
                # if original_conv.bias is not None:
                #     custom_conv.bias.data = original_conv.bias.data.clone()

                with torch.no_grad():
                    original_output = original_conv(x)
                    custom_output = custom_conv(x)

                max_diff = torch.max(torch.abs(original_output - custom_output)).item()

                if max_diff < 1e-5:
                    print(f"  ✅ Passed - Max diff: {max_diff:.2e}")
                else:
                    print(f"  ❌ Failed - Max diff: {max_diff:.2e}")

            except Exception as e:
                print(f"  💥 Error: {e}")


    print("Testing QConv2d implementation against nn.Conv2d")
    print("=" * 60)

    # Run main equivalence tests
    main_passed = test_conv_equivalence()

    # Run edge case tests
    test_edge_cases()

    # Final summary
    print(f"\n{'='*60}")
    if main_passed:
        print("🎉 Main test suite: PASSED")
    else:
        print("❌ Main test suite: FAILED")