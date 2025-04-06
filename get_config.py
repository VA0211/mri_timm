import timm
import torch

def gen_yolo_timm_config(model_name='seresnet50', input_size=640):
    model = timm.create_model(model_name, features_only=True, pretrained=False)
    dummy_input = torch.randn(1, 3, input_size, input_size)
    outputs = model(dummy_input)

    print(f"# Backbone config for timm model: {model_name}")
    print(f"- [-1, 1, Timm, [{outputs[-1].shape[1]}, '{model_name}', False, True, 0, True]]  # - 0")

    index_blocks = []
    for i, out in enumerate(outputs):
        c, h, w = out.shape[1:]
        index_blocks.append((i, c, h, w))

    # Auto-select last 3 outputs for P3, P4, P5
    selected = index_blocks[-3:]

    for idx, (i, c, h, w) in enumerate(selected):
        print(f"- [0, 1, Index, [{c}, {i}]]   # selects {i}th output (1, {c}, {w}, {h}) - {idx+1}")

    # SPPF
    sppf_in_channels = selected[-1][1]
    sppf_out_channels = sppf_in_channels // 2  # often reduced
    print(f"- [-1, 1, SPPF, [{sppf_out_channels}, 5]] # SPPF - {len(selected)+1}")

    # C2PSA
    print(f"- [-1, 2, C2PSA, [{sppf_out_channels}]]   # C2PSA - {len(selected)+2}")

gen_yolo_timm_config('convnext_tiny', input_size=640)