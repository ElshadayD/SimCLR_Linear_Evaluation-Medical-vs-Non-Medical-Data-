"""Chest X-ray pneumonia screening app (SimCLR ResNet50, fine-tuned layer4 + linear head)."""
import base64, io, os, sys
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

MODEL_PATH = os.environ.get("MODEL_PATH", "linear_eval_simclr_cxr.pt")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # ImageFolder order used in training
device = "cuda" if torch.cuda.is_available() else "cpu"


class PartialFineTuneModel(nn.Module):
    """Same structure as the training notebook so the state_dict keys match."""
    def __init__(self, num_classes=2):
        super().__init__()
        b = resnet50(weights=None)
        b.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        b.maxpool = nn.Identity()
        b.fc = nn.Identity()
        self.backbone = b
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at '{MODEL_PATH}'. Put linear_eval_simclr_cxr.pt next to app.py "
                 "or set MODEL_PATH.")
    m = PartialFineTuneModel(len(CLASS_NAMES))
    sd = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)
    return m.to(device).eval()


model = load_model()
tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Grad-CAM hooks on the last conv block
_store = {}
def _fwd(_, __, out):
    _store["act"] = out
    out.register_hook(lambda g: _store.__setitem__("grad", g))
model.backbone.layer4.register_forward_hook(_fwd)


def colorize(cam):
    """Map 0..1 array to a blue->green->yellow->red heatmap (no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(img, tta=False):
    x = tf(img).unsqueeze(0).to(device).requires_grad_(False)
    model.zero_grad()
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, 1)[0]
        cls = int(probs.argmax())
        logits[0, cls].backward()
        w = _store["grad"].mean((2, 3), keepdim=True)
        cam = F.relu((w * _store["act"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().cpu().numpy()
    if tta:  # average with horizontally flipped view for a steadier estimate
        with torch.no_grad():
            pf = torch.softmax(model(torch.flip(x, [3])), 1)[0]
        probs = (probs.detach() + pf) / 2
    probs = probs.detach().cpu().numpy()

    base = img.convert("RGB").resize((224, 224))
    heat = Image.fromarray(colorize(cam))
    return {
        "probs": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "heatmap": to_b64(heat),
        "overlay": to_b64(Image.blend(base, heat, 0.45)),
    }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", device=device)


@app.post("/api/predict")
def predict():
    tta = request.form.get("tta") == "1"
    out = []
    for f in request.files.getlist("images"):
        try:
            img = Image.open(f.stream).convert("RGB")
            res = analyze(img, tta)
            res["name"] = f.filename
            out.append(res)
        except Exception as e:  # unreadable file
            out.append({"name": f.filename, "error": f"Could not read image: {e}"})
    return jsonify(out)


if __name__ == "__main__":
    print(f"Model loaded on {device}. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
