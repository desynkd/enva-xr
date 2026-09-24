"""
Export the three stand-in networks for spike 2.

The weights are random and never trained. Spike 2 measures how long a network of a given size and
shape takes to run in the browser, and that does not depend on what the network learned.

Each network takes one 256x256 RGB picture, the size the spike 1 camera grab produces, and returns
three numbers, standing for a light direction.

    python3 tools/export-models.py
"""

import os
import torch
from torchvision import models

SIZE = 256
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")

# small is phone sized, medium is a middle point, large matches the Inception encoder used by Sial et al.
NETWORKS = {
	"small": lambda: models.mobilenet_v3_small(weights=None, num_classes=3),
	"medium": lambda: models.resnet18(weights=None, num_classes=3),
	"large": lambda: models.inception_v3(weights=None, num_classes=3, aux_logits=False, init_weights=False)
}


def export(name, build):
	model = build().eval()
	params = sum(p.numel() for p in model.parameters())
	path = os.path.join(OUT, name + ".onnx")

	torch.onnx.export(
		model,
		torch.randn(1, 3, SIZE, SIZE),
		path,
		input_names=["input"],
		output_names=["direction"],
		opset_version=17,
		do_constant_folding=True,
		dynamo=False
	)

	print(f"{name:7s} {type(model).__name__:14s} {params / 1e6:6.2f} M params   {os.path.getsize(path) / 1e6:6.1f} MB   {path}")


if __name__ == "__main__":
	os.makedirs(OUT, exist_ok=True)
	print(f"Input 1x3x{SIZE}x{SIZE}, output 1x3, random weights\n")
	for name, build in NETWORKS.items():
		export(name, build)
