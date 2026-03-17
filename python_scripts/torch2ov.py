import openvino as ov
import torch
from openvino.runtime import Core, compile_model, serialize

# load PyTorch model into memory
model = torch.jit.load("untraced/lite_walk_policy.pt")
# convert the model into OpenVINO model
example = torch.randn(750)
example2 = torch.randn(1)
ov_model = ov.convert_model(model, example_input=(example))
# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')
serialize(ov_model, "traced/lite_policy_walk.xml", "traced/lite_policy_walk.bin")
print("compiled")

