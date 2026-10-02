"""
ChurnGuard AI - Customer Churn Prediction (ANN)
================================================
Single-file Flask application for Vercel (serverless) with a full-width
horizontal form layout and integrated analytics dashboard.
"""

import os
import random
import string
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

PROJECT_NAME = "ChurnGuard AI"
TAGLINE = "Customer Retention Intelligence · ANN Demo"

# ---------------------------------------------------------------------------
# Embedded trained weights (extracted from ANN.pkl)
# ---------------------------------------------------------------------------
WEIGHTS = [
    {
        "W": np.array([[-0.0581551678, 0.2569516897, 0.0182661079, -0.0095139425, 0.2724536955, 0.0663759857, -0.1981066912, 0.5954153538], [-0.1746548712, -0.6575359702, -0.4338813722, 0.0267751552, -0.0905627906, 0.2549822628, 0.0938120782, -0.8899800777], [0.1808303148, 1.7205563784, 0.0257445816, -0.8304294348, 0.1518454701, -0.3069370687, -1.2357161045, 0.8888626695], [-0.9834831357, -0.7351076603, -0.7668297291, 1.0141049623, -0.0899414942, -0.809188962, 0.4092722535, 0.2750163674], [0.4635964334, 0.1436988562, 0.3489511609, -0.6167435646, -0.2993049324, 0.266862303, -0.6627364159, -0.1264229268], [-0.3303463757, -0.480564177, 0.5002584457, -0.1904422194, -0.5113837719, -0.3250339329, -0.1141038239, 0.6532229781], [0.9655857682, 1.5119793415, -0.3228413463, -0.5837463737, 0.933524847, -0.6232139468, -1.4442579746, 0.2388462871], [-0.4403153062, -0.2639032304, 0.4870517552, -0.0517090037, -0.3689216673, 0.2261701077, -0.0741461888, -0.8084879518], [-0.0065050479, 3.2160873413, 1.0111399889, -0.5992088914, 2.1320836544, -2.1256821156, -4.0348973274, 2.7376801968], [-5.08157e-05, -0.0003753614, 0.0331545621, 0.0490153283, -0.2083191276, 0.5802448392, 0.4935970306, -0.0582546182]], dtype="float64"),
        "b": np.array([-0.0063463431, -0.2158842236, 0.0340288691, -0.092625156, -0.0066268579, -0.0807667673, -0.0162116569, 0.0693628043], dtype="float64"),
        "act": "relu",
    },
    {
        "W": np.array([[0.5097773075, 0.1796591431, -0.2984912992, -0.4442811906, 0.2364991754, 0.576944232, -0.3914193213, -0.3642324209], [0.0720908642, -0.5596678257, 0.1914643049, 0.5295069814, 0.1406741589, 0.4544789195, -0.3793597221, -0.5270665884], [-0.417771548, -0.1856425852, 0.32053864, -0.3821976781, -0.1627099961, -0.2432026863, -0.1893026233, -0.2284141481], [-0.1949691474, 0.0351219326, 0.4449685514, 0.2955904901, -0.7024662495, 0.2507850528, 0.4425607324, -0.0539826751], [-0.5054824948, -0.0284869671, 0.3623261154, 0.813324213, -0.2539745569, 0.1837073565, -0.5617718697, -0.3210291266], [-0.1298783571, -0.1532714367, -0.0509712175, -0.631903708, 0.1018461883, -0.2426748425, -0.4937485754, -0.1960333288], [0.0092103956, 0.1294409037, -0.6643226743, -0.2341722995, -0.1853939742, -0.2227427959, 0.2415696383, -0.1293644607], [-0.0696664974, -0.349833101, -0.3397927582, 0.2232784629, 0.0431208648, 0.115854986, -0.0558219068, -0.0811071992]], dtype="float64"),
        "b": np.array([0.0567162186, -0.0340808816, -0.0471800901, 0.0437800698, -0.0503485128, 0.1597072482, -0.1433337778, 0.0], dtype="float64"),
        "act": "relu",
    },
    {
        "W": np.array([[-0.0235741176, 0.2182827592, 0.3328823745, -0.5740727782, -0.6694312692, -0.0009970099, -0.4846697748], [0.2140879333, 0.4909798205, -0.1633145958, -0.6690527201, -0.3565692902, 0.4901458919, -0.6303015351], [0.3042412698, 0.3492503166, -0.2379660755, -0.1483649015, 0.091777049, 0.2874259055, -0.2067300677], [0.7254463434, -0.3035730124, 0.2100456208, 0.2495977432, 0.0961408019, 0.8682260513, -0.7507390976], [-0.0888207555, -0.1302299351, 0.4331902266, 0.170912534, -0.0827578455, 0.2032773048, 0.4230016172], [0.0239363145, -0.009211163, -0.8428751826, 0.0642266721, -0.7152149677, 0.1111671627, -0.0228733271], [0.2300501168, -0.3010156453, 0.3926738501, -0.5995181799, 0.5873116851, 0.4397399127, 0.2305258512], [-0.2381623983, 0.0094736218, 0.1297231317, 0.2365156412, -0.3636767268, 0.2978217006, 0.5594149232]], dtype="float64"),
        "b": np.array([-0.2767629325, -0.339027673, 0.3478351235, -0.1330514252, -0.2320201248, -0.2735731006, 0.0265349355], dtype="float64"),
        "act": "relu",
    },
    {
        "W": np.array([[0.1580458879, 0.054379236, -0.4941205084, -0.08849933, 0.3768429756, 0.1291560978, -0.5337203145, 0.5957949758], [-0.4785295725, -0.1691794246, 0.342371881, -0.0566104911, 0.432915628, 0.0290739276, 0.4598015547, -0.0138657978], [-0.4284930527, 0.4761966765, -0.4212780595, -0.4383736551, -0.231090501, -0.3559965491, 0.002895494, -0.4878620207], [-0.288634032, -0.0726519674, -0.3067319691, 0.688976109, -0.4376615584, -0.098163709, 0.5369476676, 0.830468297], [-0.0044858907, -0.2908626497, -0.4914845824, 0.0135347908, -0.6166754365, -0.2566267252, -0.0087922625, -0.2561838329], [0.160170123, 0.3971165717, -0.557824254, 0.2472863346, -0.1117470339, -0.1007573977, 0.2240626663, 0.6308293939], [-0.0868995562, 0.3200522065, 0.1028306484, 0.0280627627, 0.0250909515, 0.7137650251, 0.4218035042, -0.4153401256]], dtype="float64"),
        "b": np.array([0.1325252354, 0.3230543733, 0.0, 0.6848492622, 0.6346405745, 0.0782329515, -0.3265154362, -0.7112104297], dtype="float64"),
        "act": "relu",
    },
    {
        "W": np.array([[-0.0680896044, -0.8584596515, 0.3714895546, -0.5910045505, -0.0070006819, -0.9347425103, 0.4208320677], [-0.480488956, 0.3365597725, -0.0072516897, -0.5451750755, 0.2278489619, -0.1019813567, -0.0101866527], [-0.3669618368, 0.5286855102, -0.1737235785, -0.4669302106, 0.0432223678, 0.4414945245, -0.1466300786], [0.3853541315, -0.0809229314, 0.4781853557, -0.0904031992, 0.411744684, 0.2435294986, 0.160917148], [-0.0569897071, -0.5052714348, 0.2330032587, 0.0583320186, -0.2898133397, 0.1301202476, -0.3518920839], [-0.0800909773, -0.0667366758, 0.0473137535, 0.621840477, 0.0083998889, -0.1712903678, -0.1722030193], [-0.5755434632, 0.5568862557, 0.4658168554, 0.1032427624, -0.0710925832, 0.4484865665, 0.0637277663], [-0.0293918159, 0.0587833971, -0.3565779328, 0.2824010253, 0.4582592547, 0.0900331438, -0.1459549069]], dtype="float64"),
        "b": np.array([-0.0917113051, -0.8897967935, 0.918495059, -0.695795238, 0.8469212055, -0.2898558676, -0.7148376107], dtype="float64"),
        "act": "relu",
    },
    {
        "W": np.array([[0.4810641408], [0.1066329405], [-0.2271299362], [0.0375597812], [-0.0727084652], [0.327104032], [0.6535113454]], dtype="float64"),
        "b": np.array([-0.9826385975], dtype="float64"),
        "act": "sigmoid",
    },
]

FEATURES = [
    dict(key="credit_score", label="Credit Score", section="Profile",
         kind="range", min=300, max=900, step=1, default=650, unit="",
         mean=650.53, std=96.65,
         help="Bureau credit score rating."),
    dict(key="geography", label="Geography", section="Profile",
         kind="select", options=[("0", "France"), ("1", "Germany"), ("2", "Spain")],
         default="0", mean=0.7462, std=0.8279,
         help="Country of registration."),
    dict(key="gender", label="Gender", section="Profile",
         kind="toggle2", options=[("0", "Female"), ("1", "Male")],
         default="0", mean=0.5457, std=0.4979,
         help="Gender category as encoded during training."),
    dict(key="age", label="Age", section="Profile",
         kind="range", min=18, max=92, step=1, default=35, unit=" yrs",
         mean=38.92, std=10.49,
         help="Current customer age."),
    dict(key="tenure", label="Tenure", section="Account",
         kind="range", min=0, max=10, step=1, default=5, unit=" yrs",
         mean=5.01, std=2.89,
         help="Years active as account holder."),
    dict(key="balance", label="Account Balance", section="Account",
         kind="number", min=0, max=250000, step=100, default=60000, unit="",
         mean=76485.89, std=62397.40,
         help="Total current account balance."),
    dict(key="num_products", label="Number of Products", section="Account",
         kind="stepper", min=1, max=4, step=1, default=1, unit="",
         mean=1.53, std=0.582,
         help="Total held bank products."),
    dict(key="has_cr_card", label="Has Credit Card", section="Engagement",
         kind="toggle", default="1", mean=0.7055, std=0.4558,
         help="Active credit card on file."),
    dict(key="is_active_member", label="Active Member", section="Engagement",
         kind="toggle", default="1", mean=0.5151, std=0.4998,
         help="Engaged user status."),
    dict(key="estimated_salary", label="Estimated Salary", section="Engagement",
         kind="number", min=0, max=250000, step=100, default=100000, unit="",
         mean=100090.24, std=57510.49,
         help="Estimated annual income."),
]
FEATURE_KEYS = [f["key"] for f in FEATURES]
MEANS = np.array([f["mean"] for f in FEATURES], dtype="float64")
STDS = np.array([f["std"] for f in FEATURES], dtype="float64")

SECTIONS = ["Profile", "Account", "Engagement"]

def _relu(x):
    return np.maximum(0.0, x)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

_ACTIVATIONS = {"relu": _relu, "sigmoid": _sigmoid, "linear": lambda x: x}

def run_network(x):
    for layer in WEIGHTS:
        x = x @ layer["W"] + layer["b"]
        x = _ACTIVATIONS[layer["act"]](x)
    return float(x[0])

SCALER_MEANS = None
SCALER_STDS = None
USING_REAL_SCALER = SCALER_MEANS is not None and SCALER_STDS is not None

HISTORY = []

def scale_vector(raw_vec):
    if USING_REAL_SCALER:
        return (raw_vec - SCALER_MEANS) / SCALER_STDS
    return (raw_vec - MEANS) / STDS

def predict_proba(raw_vec):
    scaled = scale_vector(raw_vec).astype("float64")
    prob = run_network(scaled)
    return max(0.0, min(1.0, prob))

def risk_bucket(prob):
    if prob < 0.30:
        return "Low", "low"
    if prob < 0.60:
        return "Medium", "medium"
    return "High", "high"

def parse_payload(payload):
    raw = np.zeros(len(FEATURES), dtype="float64")
    clean = {}
    for i, feat in enumerate(FEATURES):
        val = payload.get(feat["key"], feat["default"])
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = float(feat["default"])
        if not np.isfinite(num):
            num = float(feat["default"])
        if feat["kind"] in ("range", "number", "stepper"):
            num = max(feat.get("min", num), min(feat.get("max", num), num))
        raw[i] = num
        clean[feat["key"]] = num
    return raw, clean

def compute_impacts(raw_vec):
    base = predict_proba(raw_vec)
    impacts = []
    for i, feat in enumerate(FEATURES):
        perturbed = raw_vec.copy()
        step = STDS[i] if STDS[i] > 0 else 1.0
        perturbed[i] += step
        if feat["kind"] in ("range", "number", "stepper"):
            perturbed[i] = max(feat.get("min", perturbed[i]),
                               min(feat.get("max", perturbed[i]), perturbed[i]))
        new_p = predict_proba(perturbed)
        impacts.append({
            "key": feat["key"],
            "label": feat["label"],
            "delta": round((new_p - base) * 100, 2),
        })
    max_abs = max(1e-6, max(abs(x["delta"]) for x in impacts))
    for x in impacts:
        x["pct"] = round(abs(x["delta"]) / max_abs * 100, 1)
    impacts.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return impacts, base

def gen_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def dashboard_stats():
    total = len(HISTORY)
    if total == 0:
        return dict(total=0, churn_rate=0, avg_prob=0, high_risk=0,
                    buckets=[0] * 10, recent=[])
    probs = [h["probability"] for h in HISTORY]
    high_risk = sum(1 for p in probs if p >= 0.60)
    churned_calls = sum(1 for p in probs if p >= 0.50)
    buckets = [0] * 10
    for p in probs:
        idx = min(9, int(p * 10))
        buckets[idx] += 1
    recent = list(reversed(HISTORY[-8:]))
    return dict(
        total=total,
        churn_rate=round(churned_calls / total * 100, 1),
        avg_prob=round(sum(probs) / total * 100, 1),
        high_risk=high_risk,
        buckets=buckets,
        recent=recent,
    )

@app.route("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        project_name=PROJECT_NAME,
        tagline=TAGLINE,
        features=FEATURES,
        sections=SECTIONS,
        using_real_scaler=USING_REAL_SCALER,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    raw_vec, clean_inputs = parse_payload(payload)
    impacts, prob = compute_impacts(raw_vec)
    label, cls = risk_bucket(prob)

    record = dict(
        id=gen_id(),
        ts=datetime.now().strftime("%H:%M:%S"),
        probability=prob,
        risk=label,
        risk_class=cls,
        inputs=clean_inputs,
    )
    HISTORY.append(record)
    if len(HISTORY) > 100:
        del HISTORY[:-100]

    return jsonify(dict(
        probability=round(prob * 100, 2),
        risk=label,
        risk_class=cls,
        preprocessing_note="Estimated scaling statistics; original training scaler not supplied.",
        impacts=impacts,
        stats=dashboard_stats(),
    ))

@app.route("/api/stats")
def api_stats():
    return jsonify(dashboard_stats())

@app.route("/api/reset", methods=["POST"])
def api_reset():
    HISTORY.clear()
    return jsonify(dashboard_stats())

PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en" data-theme="midnight">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ project_name }} · Churn Risk Intelligence</title>
<style>
:root{
  --radius-lg:20px; --radius-md:14px; --radius-sm:9px;
  --ease:cubic-bezier(.22,1,.36,1);
}

/* Themes */
html[data-theme="midnight"]{
  --bg-0:#0a0d13; --bg-1:#0f131b; --bg-2:#151a24;
  --glass:rgba(255,255,255,0.045); --glass-brd:rgba(255,255,255,0.09);
  --text-hi:#eef0f4; --text-mid:#aab0bd; --text-low:#6b7280;
  --accent:#d4af5a; --accent-2:#59c2c9; --accent-ink:#20180a;
  --ok:#4ade80; --warn:#f5b942; --bad:#f2664a;
  --mesh: radial-gradient(circle at 12% 8%, rgba(212,175,90,.14), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(89,194,201,.12), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(212,175,90,.06), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
html[data-theme="emerald"]{
  --bg-0:#071410; --bg-1:#0a1a15; --bg-2:#0f221c;
  --glass:rgba(160,255,210,0.045); --glass-brd:rgba(160,255,210,0.09);
  --text-hi:#eafaf1; --text-mid:#9fc4b3; --text-low:#5f8272;
  --accent:#34d399; --accent-2:#c084fc; --accent-ink:#04140d;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 10% 10%, rgba(52,211,153,.14), transparent 42%),
          radial-gradient(circle at 90% 15%, rgba(192,132,252,.10), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(52,211,153,.06), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
html[data-theme="amethyst"]{
  --bg-0:#0f0a1a; --bg-1:#150e22; --bg-2:#1b122b;
  --glass:rgba(216,180,254,0.05); --glass-brd:rgba(216,180,254,0.10);
  --text-hi:#f3ecfb; --text-mid:#b9a8cf; --text-low:#786690;
  --accent:#a855f7; --accent-2:#f472b6; --accent-ink:#180a26;
  --ok:#4ade80; --warn:#f5b942; --bad:#fb7185;
  --mesh: radial-gradient(circle at 12% 10%, rgba(168,85,247,.16), transparent 42%),
          radial-gradient(circle at 88% 14%, rgba(244,114,182,.12), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(168,85,247,.07), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}
html[data-theme="ivory"]{
  --bg-0:#e7edf1; --bg-1:#eef2f5; --bg-2:#f5f7f9;
  --glass:rgba(255,255,255,0.55); --glass-brd:rgba(27,42,74,0.10);
  --text-hi:#16202e; --text-mid:#3c4a5c; --text-low:#748094;
  --accent:#1b2a4a; --accent-2:#c97b4a; --accent-ink:#f5f7f9;
  --ok:#1a9e6b; --warn:#b5790f; --bad:#c23b32;
  --mesh: radial-gradient(circle at 12% 8%, rgba(27,42,74,.07), transparent 42%),
          radial-gradient(circle at 88% 12%, rgba(201,123,74,.10), transparent 40%),
          radial-gradient(circle at 50% 100%, rgba(27,42,74,.04), transparent 55%);
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype',serif;
}

*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background: var(--mesh), linear-gradient(180deg, var(--bg-0), var(--bg-1) 45%, var(--bg-0));
  background-attachment:fixed;
  color:var(--text-hi);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
  transition:background .5s var(--ease), color .5s var(--ease);
}
.wrap{max-width:1200px; margin:0 auto; padding:28px 24px 80px;}

.topbar{display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:28px; flex-wrap:wrap;}
.brand{display:flex; align-items:center; gap:12px;}
.brand-mark{
  width:42px;height:42px;border-radius:12px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 6px 18px -6px rgba(0,0,0,.5);
}
.brand-mark svg{width:22px;height:22px;}
.brand-text h1{font-family:var(--font-display); font-weight:700; font-size:22px; margin:0; color:var(--text-hi);}
.brand-text p{margin:2px 0 0; font-size:12.5px; color:var(--text-mid);}

.theme-switch{display:flex; gap:8px; background:var(--glass); border:1px solid var(--glass-brd); padding:6px; border-radius:999px; backdrop-filter:blur(14px);}
.theme-dot{width:30px;height:30px;border-radius:50%; cursor:pointer; border:2px solid transparent; transition:transform .25s var(--ease);}
.theme-dot:hover{transform:scale(1.1);}
.theme-dot.active{border-color:var(--text-hi);}
.theme-dot[data-t="midnight"]{background:linear-gradient(135deg,#0f131b,#d4af5a);}
.theme-dot[data-t="emerald"]{background:linear-gradient(135deg,#0a1a15,#34d399);}
.theme-dot[data-t="amethyst"]{background:linear-gradient(135deg,#150e22,#a855f7);}
.theme-dot[data-t="ivory"]{background:linear-gradient(135deg,#eef2f5,#1b2a4a);}

.card{
  background:var(--glass); border:1px solid var(--glass-brd);
  border-radius:var(--radius-lg); backdrop-filter:blur(18px);
  box-shadow:0 20px 50px -30px rgba(0,0,0,.6);
  padding:26px; margin-bottom:24px;
}

/* Horizontal Form Grid Layout */
.form-card h2{font-family:var(--font-display); font-size:20px; margin:0 0 4px;}
.form-card .sub{color:var(--text-mid); font-size:13px; margin:0 0 20px;}

.section-label{
  font-size:11.5px; letter-spacing:1.5px; text-transform:uppercase; color:var(--accent);
  margin:20px 0 12px; font-weight:700; border-bottom:1px solid var(--glass-brd); padding-bottom:6px;
}
.section-label:first-of-type{margin-top:0;}

.horizontal-fields{
  display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:18px 22px; margin-bottom:16px;
}

.field{display:flex; flex-direction:column; justify-content:space-between;}
.field-head{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;}
.field-head label{font-size:13px; color:var(--text-hi); font-weight:600;}
.field-head .val{font-size:12.5px; color:var(--accent); font-weight:600;}
.field .help{font-size:11.5px; color:var(--text-low); margin-top:4px;}

input[type="range"]{
  -webkit-appearance:none; width:100%; height:6px; border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); outline:none; cursor:pointer;
}
input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none; width:18px; height:18px; border-radius:50%;
  background:var(--text-hi); border:3px solid var(--accent); cursor:pointer;
}

input[type="number"], select{
  width:100%; padding:9px 12px; border-radius:var(--radius-sm);
  border:1px solid var(--glass-brd); background:rgba(0,0,0,.15);
  color:var(--text-hi); font-size:13.5px; outline:none;
}
html[data-theme="ivory"] input[type="number"], html[data-theme="ivory"] select{background:rgba(255,255,255,.6);}

.seg{display:flex; border-radius:var(--radius-sm); overflow:hidden; border:1px solid var(--glass-brd);}
.seg button{
  flex:1; padding:8px; background:rgba(0,0,0,.12); color:var(--text-mid);
  border:none; font-size:12.5px; font-weight:600; cursor:pointer;
}
.seg button.active{background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:var(--accent-ink);}

.toggle-row{display:flex; align-items:center; justify-content:space-between; min-height:42px;}
.switch{position:relative; width:44px; height:24px;}
.switch input{opacity:0; width:0; height:0;}
.slider-pill{
  position:absolute; inset:0; background:rgba(0,0,0,.25); border-radius:999px; cursor:pointer; border:1px solid var(--glass-brd);
}
.slider-pill:before{
  content:""; position:absolute; width:18px; height:18px; left:2px; top:2px;
  background:var(--text-hi); border-radius:50%; transition:transform .2s var(--ease);
}
.switch input:checked + .slider-pill{background:linear-gradient(135deg,var(--accent),var(--accent-2));}
.switch input:checked + .slider-pill:before{transform:translateX(20px);}

.stepper{display:flex; align-items:center; gap:8px;}
.stepper button{
  width:32px;height:32px;border-radius:50%; border:1px solid var(--glass-brd);
  background:rgba(0,0,0,.15); color:var(--text-hi); font-size:15px; cursor:pointer;
}
.stepper .count{flex:1; text-align:center; font-size:15px; font-weight:700;}

.actions{display:flex; gap:12px; margin-top:20px; justify-content:flex-end;}
.btn{
  border:none; border-radius:999px; font-weight:700; font-size:14px; cursor:pointer;
  padding:12px 26px; display:inline-flex; align-items:center; gap:8px; transition:transform .2s var(--ease);
}
.btn-primary{
  color:var(--accent-ink); background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 8px 24px -8px var(--accent);
}
.btn-primary:hover{transform:translateY(-2px);}
.btn-ghost{background:transparent; color:var(--text-mid); border:1px solid var(--glass-brd);}

/* Assessment Section */
.result-grid{display:grid; grid-template-columns:300px 1fr; gap:24px; align-items:center;}
@media(max-width:768px){.result-grid{grid-template-columns:1fr; text-align:center;}}
.gauge-wrap{position:relative; width:220px; margin:0 auto;}
.gauge-num{position:absolute; left:0; right:0; top:58%; transform:translateY(-50%); font-family:var(--font-display); font-size:36px; font-weight:700; text-align:center;}
.gauge-label{position:absolute; left:0; right:0; top:80%; font-size:11px; text-transform:uppercase; color:var(--text-mid); text-align:center;}
.risk-pill{display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px; font-size:12px; font-weight:700; margin-top:10px;}
.risk-pill.low{background:rgba(74,222,128,.15); color:var(--ok);}
.risk-pill.medium{background:rgba(245,185,66,.15); color:var(--warn);}
.risk-pill.high{background:rgba(242,102,74,.15); color:var(--bad);}
.risk-pill .dot{width:6px;height:6px;border-radius:50%; background:currentColor;}

.impacts h3{font-size:11.5px; letter-spacing:1px; text-transform:uppercase; color:var(--text-mid); margin:0 0 10px;}
.impact-row{margin-bottom:8px;}
.impact-row .top{display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;}
.impact-track{height:6px; border-radius:6px; background:rgba(0,0,0,.2); overflow:hidden;}
.impact-fill{height:100%; border-radius:6px; transition:width .5s var(--ease);}
.impact-fill.up{background:linear-gradient(90deg,var(--bad),#ff9d80);}
.impact-fill.down{background:linear-gradient(90deg,var(--ok),#8bf5b0);}

/* Stats Grid */
.stat-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:20px;}
@media(max-width:640px){.stat-grid{grid-template-columns:repeat(2,1fr);}}
.stat-card{padding:16px 20px;}
.stat-card .label{font-size:11px; text-transform:uppercase; color:var(--text-mid); margin-bottom:6px;}
.stat-card .value{font-family:var(--font-display); font-size:26px; font-weight:700;}

.dash-grid{display:grid; grid-template-columns:1.2fr 1fr; gap:18px;}
@media(max-width:700px){.dash-grid{grid-template-columns:1fr;}}
.hist-row{display:flex; align-items:flex-end; gap:6px; height:110px;}
.hist-bar{flex:1; background:linear-gradient(180deg,var(--accent),var(--accent-2)); border-radius:4px 4px 2px 2px; min-height:3px; transition:height .4s var(--ease);}
.hist-labels{display:flex; gap:6px; margin-top:6px;}
.hist-labels span{flex:1; text-align:center; font-size:9.5px; color:var(--text-low);}

table.recent{width:100%; border-collapse:collapse; font-size:12.5px;}
table.recent th{text-align:left; color:var(--text-mid); font-size:10px; text-transform:uppercase; padding:0 8px 8px 0;}
table.recent td{padding:8px 8px 8px 0; border-top:1px solid var(--glass-brd);}
table.recent .badge{padding:3px 8px; border-radius:999px; font-size:10.5px; font-weight:700;}
table.recent .badge.low{background:rgba(74,222,128,.15); color:var(--ok);}
table.recent .badge.medium{background:rgba(245,185,66,.15); color:var(--warn);}
table.recent .badge.high{background:rgba(242,102,74,.15); color:var(--bad);}

footer{margin-top:30px; text-align:center; color:var(--text-low); font-size:11.5px;}
</style>
</head>
<body data-theme="midnight">
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none"><path d="M12 2L3 6v6c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6l-9-4z" fill="var(--accent-ink)"/><path d="M8.5 12l2.3 2.3L16 9" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="brand-text">
        <h1>{{ project_name }}</h1>
        <p>{{ tagline }}</p>
      </div>
    </div>
    <div class="theme-switch" id="themeSwitch">
      <div class="theme-dot active" data-t="midnight" title="Midnight Gold"></div>
      <div class="theme-dot" data-t="emerald" title="Emerald Vault"></div>
      <div class="theme-dot" data-t="amethyst" title="Royal Amethyst"></div>
      <div class="theme-dot" data-t="ivory" title="Arctic Ivory"></div>
    </div>
  </div>

  <!-- FULL-WIDTH HORIZONTAL FORM -->
  <div class="card form-card">
    <h2>Customer Profile Information</h2>
    <p class="sub">Enter account and engagement metrics below to evaluate risk.</p>
    <form id="predictForm">
      {% for sec in sections %}
      <div class="section-label">{{ sec }} Details</div>
      <div class="horizontal-fields">
        {% for f in features if f.section == sec %}
          <div class="field" data-key="{{ f.key }}">
            {% if f.kind == "range" %}
              <div class="field-head">
                <label for="in_{{ f.key }}">{{ f.label }}</label>
                <span class="val" id="val_{{ f.key }}">{{ f.default }}{{ f.unit }}</span>
              </div>
              <input type="range" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "number" %}
              <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
              <input type="number" id="in_{{ f.key }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" value="{{ f.default }}">
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "select" %}
              <div class="field-head"><label for="in_{{ f.key }}">{{ f.label }}</label></div>
              <select id="in_{{ f.key }}">
                {% for val, txt in f.options %}
                <option value="{{ val }}" {% if val == f.default %}selected{% endif %}>{{ txt }}</option>
                {% endfor %}
              </select>
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "toggle2" %}
              <div class="field-head"><label>{{ f.label }}</label></div>
              <div class="seg" id="in_{{ f.key }}" data-value="{{ f.default }}">
                {% for val, txt in f.options %}
                <button type="button" data-v="{{ val }}" class="{% if val == f.default %}active{% endif %}">{{ txt }}</button>
                {% endfor %}
              </div>
              <div class="help">{{ f.help }}</div>

            {% elif f.kind == "toggle" %}
              <div class="toggle-row">
                <div>
                  <div class="field-head" style="margin-bottom:0;"><label>{{ f.label }}</label></div>
                  <div class="help">{{ f.help }}</div>
                </div>
                <label class="switch">
                  <input type="checkbox" id="in_{{ f.key }}" {% if f.default == "1" %}checked{% endif %}>
                  <span class="slider-pill"></span>
                </label>
              </div>

            {% elif f.kind == "stepper" %}
              <div class="field-head"><label>{{ f.label }}</label></div>
              <div class="stepper" id="in_{{ f.key }}" data-value="{{ f.default }}" data-min="{{ f.min }}" data-max="{{ f.max }}">
                <button type="button" data-d="-1">−</button>
                <div class="count">{{ f.default }}</div>
                <button type="button" data-d="1">+</button>
              </div>
              <div class="help">{{ f.help }}</div>
            {% endif %}
          </div>
        {% endfor %}
      </div>
      {% endfor %}

      <div class="actions">
        <button type="button" class="btn btn-ghost" id="resetBtn">Reset</button>
        <button type="submit" class="btn btn-primary" id="predictBtn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M13 2L3 14h8l-1 8 11-14h-8l1-6z" fill="currentColor"/></svg>
          Run Churn Assessment
        </button>
      </div>
    </form>
  </div>

  <!-- RESULT / GAUGE -->
  <div class="card">
    <div class="result-grid">
      <div style="display:flex; flex-direction:column; align-items:center;">
        <div class="gauge-wrap">
          <svg id="gaugeSvg" viewBox="0 0 240 140" width="220" height="130"></svg>
          <div class="gauge-num" id="gaugeNum">--%</div>
          <div class="gauge-label" id="gaugeLbl">awaiting input</div>
        </div>
        <div class="risk-pill low" id="riskPill" style="visibility:hidden;">
          <span class="dot"></span><span id="riskText">Low Risk</span>
        </div>
      </div>
      <div>
        <div id="placeholderNote" style="color:var(--text-low); font-size:13px; line-height:1.6;">
          Configure customer inputs and select <b>Run Churn Assessment</b> to calculate live model churn probability and see top sensitivity drivers.
        </div>
        <div class="impacts" id="impactsBlock" style="display:none;">
          <h3>Top Sensitivity Influencers</h3>
          <div id="impactsList"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ANALYTICS DASHBOARD -->
  <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:12px;">
    <h2 style="font-family:var(--font-display); font-size:18px; margin:0;">In-Memory Run Telemetry</h2>
    <button id="clearHistoryBtn" style="background:none; border:none; color:var(--text-mid); font-size:12px; cursor:pointer; text-decoration:underline;">Clear telemetry</button>
  </div>

  <div class="stat-grid">
    <div class="card stat-card"><div class="label">Total Runs</div><div class="value" id="statTotal">0</div></div>
    <div class="card stat-card"><div class="label">Churn Rate</div><div class="value" id="statChurn">0<span>%</span></div></div>
    <div class="card stat-card"><div class="label">Avg. Probability</div><div class="value" id="statAvg">0<span>%</span></div></div>
    <div class="card stat-card"><div class="label">High Risk Alerts</div><div class="value" id="statHigh">0</div></div>
  </div>

  <div class="dash-grid">
    <div class="card" style="margin-bottom:0;">
      <h3 style="font-size:12.5px; margin:0 0 14px; text-transform:uppercase; letter-spacing:.8px; color:var(--text-mid);">Probability Distribution</h3>
      <div class="hist-row" id="histRow"></div>
      <div class="hist-labels" id="histLabels"></div>
    </div>
    <div class="card" style="margin-bottom:0;">
      <h3 style="font-size:12.5px; margin:0 0 14px; text-transform:uppercase; letter-spacing:.8px; color:var(--text-mid);">Recent Executions</h3>
      <div id="recentWrap"><div style="color:var(--text-low); font-size:12px; text-align:center; padding:16px;">No recorded predictions yet.</div></div>
    </div>
  </div>

  <footer>
    Demonstration only: original fitted preprocessing scaler was not provided. Predictions may differ from the trained model. In-memory statistics are temporary on serverless hosting.<br>
    Model: Sequential ANN · Dense(8→8→7→8→7→1) · Sigmoid Output · 10 Features
  </footer>
</div>

<script>
const FEATURE_META = {{ features | tojson }};

const themeSwitch = document.getElementById('themeSwitch');
themeSwitch.addEventListener('click', (e) => {
  const dot = e.target.closest('.theme-dot');
  if(!dot) return;
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  dot.classList.add('active');
  document.documentElement.setAttribute('data-theme', dot.dataset.t);
});

FEATURE_META.forEach(f => {
  if(f.kind === 'range'){
    const el = document.getElementById('in_' + f.key);
    const out = document.getElementById('val_' + f.key);
    el.addEventListener('input', () => out.textContent = el.value + (f.unit || ''));
  }
  if(f.kind === 'toggle2'){
    const wrap = document.getElementById('in_' + f.key);
    wrap.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        wrap.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        wrap.dataset.value = btn.dataset.v;
      });
    });
  }
  if(f.kind === 'stepper'){
    const wrap = document.getElementById('in_' + f.key);
    const countEl = wrap.querySelector('.count');
    wrap.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        let v = parseInt(wrap.dataset.value, 10) + parseInt(btn.dataset.d, 10);
        v = Math.max(parseInt(wrap.dataset.min,10), Math.min(parseInt(wrap.dataset.max,10), v));
        wrap.dataset.value = v;
        countEl.textContent = v;
      });
    });
  }
});

function collectPayload(){
  const payload = {};
  FEATURE_META.forEach(f => {
    if(f.kind === 'range' || f.kind === 'number'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).value);
    } else if(f.kind === 'select'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).value);
    } else if(f.kind === 'toggle2' || f.kind === 'stepper'){
      payload[f.key] = parseFloat(document.getElementById('in_' + f.key).dataset.value);
    } else if(f.kind === 'toggle'){
      payload[f.key] = document.getElementById('in_' + f.key).checked ? 1 : 0;
    }
  });
  return payload;
}

function drawGauge(pct){
  const svg = document.getElementById('gaugeSvg');
  const cx = 120, cy = 115, r = 85;
  const color = pct < 30 ? 'var(--ok)' : pct < 60 ? 'var(--warn)' : 'var(--bad)';
  const endAngle = Math.PI - (pct/100)*Math.PI;
  const polar = (ang) => [cx + r*Math.cos(ang), cy - r*Math.sin(ang)];
  const trackStart = polar(Math.PI), trackEnd = polar(0);
  const valEnd = polar(endAngle);
  const largeArc = pct > 50 ? 1 : 0;

  svg.innerHTML = `
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 1 1 ${trackEnd[0]} ${trackEnd[1]}"
          fill="none" stroke="rgba(120,120,140,0.18)" stroke-width="14" stroke-linecap="round"/>
    <path d="M ${trackStart[0]} ${trackStart[1]} A ${r} ${r} 0 ${largeArc} 1 ${valEnd[0]} ${valEnd[1]}"
          fill="none" stroke="${color}" stroke-width="14" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="4" fill="${color}"/>
  `;
  document.getElementById('gaugeNum').textContent = pct.toFixed(1) + '%';
  document.getElementById('gaugeNum').style.color = color;
  document.getElementById('gaugeLbl').textContent = 'churn probability';
}

const form = document.getElementById('predictForm');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  try{
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(collectPayload())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction request failed");
    renderResult(data);
    renderDashboard(data.stats);
  } catch(err){
    console.error(err);
    alert("Prediction failed. Please check the deployment logs and try again.");
  } finally {
    btn.disabled = false;
  }
});

function renderResult(data){
  document.getElementById('placeholderNote').style.display = 'none';
  drawGauge(data.probability);

  const pill = document.getElementById('riskPill');
  pill.style.visibility = 'visible';
  pill.className = 'risk-pill ' + data.risk_class;
  document.getElementById('riskText').textContent = data.risk + ' Risk';

  const impactsBlock = document.getElementById('impactsBlock');
  const list = document.getElementById('impactsList');
  list.innerHTML = '';
  data.impacts.slice(0, 4).forEach(im => {
    const dir = im.delta >= 0 ? 'up' : 'down';
    const row = document.createElement('div');
    row.className = 'impact-row';
    row.innerHTML = `
      <div class="top">
        <span>${im.label}</span>
        <span>${im.delta > 0 ? '+' : ''}${im.delta.toFixed(2)} pp</span>
      </div>
      <div class="impact-track"><div class="impact-fill ${dir}" style="width:${im.pct}%"></div></div>
    `;
    list.appendChild(row);
  });
  impactsBlock.style.display = 'block';
}

function renderDashboard(stats){
  document.getElementById('statTotal').textContent = stats.total;
  document.getElementById('statChurn').innerHTML = stats.churn_rate + '<span>%</span>';
  document.getElementById('statAvg').innerHTML = stats.avg_prob + '<span>%</span>';
  document.getElementById('statHigh').textContent = stats.high_risk;

  const histRow = document.getElementById('histRow');
  const histLabels = document.getElementById('histLabels');
  histRow.innerHTML = '';
  histLabels.innerHTML = '';
  const maxCount = Math.max(1, ...stats.buckets);
  stats.buckets.forEach((c, i) => {
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = Math.max(3, (c / maxCount) * 100) + 'px';
    histRow.appendChild(bar);
    const lbl = document.createElement('span');
    lbl.textContent = i % 2 === 0 ? (i*10) + '%' : '';
    histLabels.appendChild(lbl);
  });

  const recentWrap = document.getElementById('recentWrap');
  if(stats.recent.length === 0){
    recentWrap.innerHTML = '<div style="color:var(--text-low); font-size:12px; text-align:center; padding:16px;">No recorded predictions yet.</div>';
    return;
  }
  let rows = stats.recent.map(r => `
    <tr>
      <td>${r.ts}</td>
      <td>${(r.probability*100).toFixed(1)}%</td>
      <td><span class="badge ${r.risk_class}">${r.risk}</span></td>
    </tr>
  `).join('');
  recentWrap.innerHTML = `
    <table class="recent">
      <thead><tr><th>Time</th><th>Probability</th><th>Risk</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

document.getElementById('resetBtn').addEventListener('click', () => window.location.reload());
document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
  const res = await fetch('/api/reset', {method:'POST'});
  const stats = await res.json();
  renderDashboard(stats);
});

drawGauge(0);
document.getElementById('gaugeNum').textContent = '--%';
document.getElementById('gaugeLbl').textContent = 'awaiting input';
fetch('/api/stats').then(r => r.json()).then(renderDashboard);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
