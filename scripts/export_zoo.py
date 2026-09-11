#!/usr/bin/env python3
"""Export one model from the zoo to ONNX at a stated input size.

Covers the non-YOLO half of the sweep. ultralytics models are exported by
scripts/export_one.sh; everything here comes from torchvision or transformers.

Usage: python3 export_zoo.py <name> <out_dir>
"""
import os
import sys

import torch

NAME, OUT = sys.argv[1], sys.argv[2]


def tv(fn, size, **kw):
    """torchvision scatters model builders across three submodules."""
    import torchvision.models as M
    import torchvision.models.detection as D
    import torchvision.models.segmentation as S
    for mod in (M, S, D):
        if hasattr(mod, fn):
            return getattr(mod, fn)(weights="DEFAULT", **kw).eval().cuda(), size
    raise AttributeError("no torchvision builder named %r" % fn)


class DetWrap(torch.nn.Module):
    """torchvision detection models take a LIST of images and return a list of
    dicts. ONNX wants tensors in and tensors out, so unpack to the three heads."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        r = self.m([x[0]])[0]
        return r["boxes"], r["scores"], r["labels"]


class SegWrap(torch.nn.Module):
    """torchvision segmentation models return a dict; ONNX wants a tensor."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]



class HFWrap(torch.nn.Module):
    """HuggingFace vision models return a ModelOutput dataclass; take the first
    tensor field so ONNX sees a plain tensor."""
    def __init__(self, m, field):
        super().__init__()
        self.m, self.field = m, field

    def forward(self, x):
        return getattr(self.m(pixel_values=x), self.field)


def hf_seg(repo, size):
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained(repo)
    return HFWrap(m, "logits"), size


def hf_depth(repo, size):
    from transformers import AutoModelForDepthEstimation
    return HFWrap(AutoModelForDepthEstimation.from_pretrained(repo), "predicted_depth"), size


def hf_backbone(repo, size):
    from transformers import AutoModel
    return HFWrap(AutoModel.from_pretrained(repo), "last_hidden_state"), size


def unet(encoder, size):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=19), size


def sam_encoder(repo, size=1024):
    """SAM is two engines: a vision encoder that runs ONCE PER FRAME and a mask
    decoder that runs once per prompt. Only the encoder is comparable to the
    other per-frame costs in this sweep, so that is what we export. ultralytics
    cannot export SAM to ONNX at all (checkpoint format / missing .args), hence
    transformers."""
    from transformers import SamModel
    m = SamModel.from_pretrained(repo).vision_encoder
    class Enc(torch.nn.Module):
        def __init__(self, e):
            super().__init__()
            self.e = e
        def forward(self, x):
            out = self.e(x)
            return out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    return Enc(m), size


ZOO = {
    "resnet50":          lambda: tv("resnet50", 224),
    "efficientnet_b0":   lambda: tv("efficientnet_b0", 224),
    "efficientnet_v2_s": lambda: tv("efficientnet_v2_s", 384),
    "deeplabv3_mnv3":    lambda: (SegWrap(tv("deeplabv3_mobilenet_v3_large", 640)[0]), 640),
    "segformer_b0":      lambda: hf_seg("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 512),
    "segformer_b2":      lambda: hf_seg("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", 512),
    "segformer_b5":      lambda: hf_seg("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", 512),
    "unet_r34":          lambda: unet("resnet34", 640),
    "depth_anything_v2_l": lambda: hf_depth("depth-anything/Depth-Anything-V2-Large-hf", 518),
    "dinov2_l":          lambda: hf_backbone("facebook/dinov2-large", 518),
    "sam_vit_b_enc":     lambda: sam_encoder("facebook/sam-vit-base"),
    "sam_vit_h_enc":     lambda: sam_encoder("facebook/sam-vit-huge"),
    "ssdlite_mnv3":      lambda: (DetWrap(tv("ssdlite320_mobilenet_v3_large", 320)[0]), 320),
    "maskrcnn_r50":      lambda: (DetWrap(tv("maskrcnn_resnet50_fpn", 800)[0]), 800),
}

if NAME not in ZOO:
    sys.exit("unknown model %r; have: %s" % (NAME, " ".join(sorted(ZOO))))

model, size = ZOO[NAME]()
model = model.eval().cuda()
x = torch.randn(1, 3, size, size, device="cuda")

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, NAME + ".onnx")
torch.onnx.export(model, (x,), path, input_names=["images"], output_names=["output0"] if not isinstance(model, DetWrap)
                  else ["boxes", "scores", "labels"],
                  opset_version=17, dynamo=False)
print("%s  input=1x3x%dx%d  -> %s" % (NAME, size, size, path))
