"""
Customer Churn Prediction (ANN)
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

HISTORY = []  # Temporary per-instance telemetry; not persistent on Vercel

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
<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ChurnScope | ANN Intelligence Studio</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,'Segoe UI',sans-serif;color-scheme:dark;--bg:#0b1020;--side:#10192d;--panel:#151f35;--panel2:#1b2942;--line:#2a3850;--text:#f2f6ff;--muted:#99a9c4;--primary:#7c9cff;--secondary:#54dfc1;--danger:#ff728b;--warning:#ffca73;--shadow:0 16px 40px #0002}
html[data-theme=light]{color-scheme:light;--bg:#f1f5fb;--side:#fff;--panel:#fff;--panel2:#f5f8ff;--line:#dfe6f0;--text:#17233b;--muted:#66758e;--primary:#4d65df;--secondary:#087f74;--danger:#d83f64;--warning:#ad7000;--shadow:0 12px 35px #20345b12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}button,input,select{font:inherit}button{cursor:pointer}button:disabled{opacity:.6;cursor:wait}.layout{display:grid;grid-template-columns:245px minmax(0,1fr);min-height:100vh}.sidebar{background:var(--side);border-right:1px solid var(--line);padding:27px 18px;display:flex;flex-direction:column;gap:30px;position:sticky;top:0;height:100vh}.logo{display:flex;align-items:center;gap:12px;font-size:18px;font-weight:850;letter-spacing:-.5px}.logo-icon{width:39px;height:39px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(135deg,#7289ff,#36cdb3);color:#fff}.eyebrow{font-size:11px;letter-spacing:1.7px;text-transform:uppercase;color:var(--muted);font-weight:800}.nav{display:grid;gap:8px}.nav a{color:var(--muted);text-decoration:none;padding:13px;border-radius:11px;font-size:13px;font-weight:700}.nav a:hover,.nav a.active{background:var(--panel2);color:var(--primary)}.sidebar-foot{margin-top:auto;font-size:12px;color:var(--muted);line-height:1.7}.main{padding:28px clamp(18px,3vw,46px) 70px;max-width:1640px;width:100%;margin:auto}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:25px}.top small{color:var(--muted)}.top-right{display:flex;align-items:center;gap:10px}.status{color:var(--secondary);background:var(--panel);border:1px solid var(--line);padding:10px 14px;border-radius:30px;font-size:12px}.btn{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:11px 17px;border-radius:11px;font-weight:750}.btn.primary{background:var(--primary);color:#fff;border-color:transparent}.btn:hover{filter:brightness(1.1)}.hero{background:linear-gradient(110deg,#23346b,#19294e 60%,#154b54);padding:32px;border-radius:20px;display:flex;justify-content:space-between;gap:20px;align-items:center;box-shadow:var(--shadow);margin-bottom:22px;color:#fff}.hero h1{font-size:clamp(25px,3vw,38px);margin:10px 0;letter-spacing:-1.1px}.hero p{color:#c4d5ef;margin:0;max-width:620px;line-height:1.6}.hero-symbol{font-size:74px;opacity:.65}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin-bottom:22px}.card{background:var(--panel);border:1px solid var(--line);border-radius:17px;padding:23px;box-shadow:var(--shadow)}.kpi .label{font-size:12px;color:var(--muted)}.kpi strong{font-size:30px;display:block;margin:12px 0 6px;letter-spacing:-1px}.kpi small{color:var(--muted)}.content{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.85fr);gap:20px}.heading{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:19px}.heading h2{font-size:17px;margin:0}.heading span{font-size:12px;color:var(--muted)}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 22px}.tab{padding:10px 13px;border-radius:9px;border:1px solid var(--line);background:var(--panel2);color:var(--muted);font-weight:750;font-size:12px}.tab.active{background:var(--primary);color:#fff;border-color:var(--primary)}.formgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px}.field label{display:block;font-size:12px;font-weight:800;margin-bottom:8px}.field small{display:block;color:var(--muted);font-size:11px;margin-top:6px;line-height:1.4}.field input,.field select{width:100%;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:12px;border-radius:10px;outline:none}.field input:focus,.field select:focus{border-color:var(--primary)}.field input[type=range]{accent-color:var(--primary);padding:0}.range-head{display:flex;justify-content:space-between}.range-head output{color:var(--primary)}.actions{display:flex;justify-content:flex-end;gap:10px;margin-top:27px}.result{display:grid;gap:20px}.donut{--pct:0;--ring:var(--primary);width:210px;height:210px;border-radius:50%;background:conic-gradient(var(--ring) calc(var(--pct)*1%),var(--line) 0);display:grid;place-items:center;margin:15px auto;position:relative}.donut:before{content:'';position:absolute;inset:18px;border-radius:50%;background:var(--panel)}.donut-inner{z-index:1;text-align:center}.donut-inner strong{font-size:39px;letter-spacing:-1.5px}.donut-inner small{display:block;color:var(--muted)}.risk{text-align:center;font-size:14px;font-weight:800}.note{color:var(--muted);font-size:12px;line-height:1.7}.driver{margin:15px 0}.driver-top{display:flex;justify-content:space-between;gap:8px;font-size:12px;margin-bottom:7px}.track{height:8px;border-radius:20px;background:var(--line);overflow:hidden}.fill{height:100%;border-radius:20px;background:var(--primary)}.bottom{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}.bars{display:flex;gap:9px;align-items:end;height:160px;padding-top:15px}.bar{flex:1;min-height:3px;background:linear-gradient(var(--primary),var(--secondary));border-radius:6px 6px 0 0}.axis{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:8px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:12px}th{color:var(--muted);font-weight:700}td,th{padding:11px 7px;border-bottom:1px solid var(--line)}.pill{display:inline-block;padding:5px 8px;border-radius:8px;background:var(--panel2)}.alert{display:none;margin-top:15px;padding:12px;border-radius:10px;background:#ff728b20;color:var(--danger);font-size:13px}.foot{margin-top:22px;color:var(--muted);font-size:11px;line-height:1.7}.hidden{display:none!important}@media(max-width:1100px){.content{grid-template-columns:1fr}.result{grid-template-columns:1fr 1fr}.kpis{grid-template-columns:repeat(2,1fr)}}@media(max-width:750px){.layout{display:block}.sidebar{height:auto;position:static;padding:14px 18px}.sidebar .nav,.sidebar-foot{display:none}.main{padding:18px}.hero-symbol{display:none}.hero{padding:24px}.bottom,.result{grid-template-columns:1fr}}@media(max-width:480px){.formgrid{grid-template-columns:1fr}.kpis{gap:9px}.card{padding:16px}.kpi strong{font-size:24px}}
</style></head><body><div class="layout"><aside class="sidebar"><div class="logo"><span class="logo-icon">◈</span> ChurnScope <span style="color:var(--secondary)">AI</span></div><div><div class="eyebrow" style="padding:0 12px 13px">Workspace</div><nav class="nav"><a class="active" href="#overview">▦ &nbsp; Overview</a><a href="#assessment">◉ &nbsp; Risk Assessment</a><a href="#analytics">▥ &nbsp; Analytics</a><a href="#history">◷ &nbsp; Recent Runs</a></nav></div><div class="sidebar-foot">ANN Prediction Studio<br>10 input features · 6 dense layers<br><br>Demo predictions only</div></aside><main class="main" id="overview"><div class="top"><div><div class="eyebrow">Customer intelligence / workspace</div><small>Interactive churn modeling dashboard</small></div><div class="top-right"><span class="status">● &nbsp; Model online</span><button class="btn" id="themeBtn" type="button">☼ Theme</button></div></div><section class="hero"><div><div class="eyebrow" style="color:#a9d5ef">Prediction intelligence platform</div><h1>Customer Churn Analytics</h1><p>Explore customer retention risk with an artificial neural network. Adjust customer details, run a prediction, and review scenario sensitivity.</p></div><div class="hero-symbol">◉</div></section><section class="kpis"><div class="card kpi"><div class="label">Predictions in this instance</div><strong id="kTotal">0</strong><small>Temporary session telemetry</small></div><div class="card kpi"><div class="label">Average predicted risk</div><strong id="kAvg">—</strong><small>Across recorded predictions</small></div><div class="card kpi"><div class="label">High-risk predictions</div><strong id="kHigh">0</strong><small>Probability ≥ 60%</small></div><div class="card kpi"><div class="label">Predicted churn classifications</div><strong id="kChurn">—</strong><small>Threshold ≥ 50%; not actual churn</small></div></section><section class="content"><article class="card" id="assessment"><div class="heading"><h2>Customer risk assessment</h2><span>10 model features</span></div><div class="tabs" id="tabs"><button type="button" class="tab active" data-tab="Profile">01 · Customer</button><button type="button" class="tab" data-tab="Account">02 · Banking</button><button type="button" class="tab" data-tab="Engagement">03 · Engagement</button></div><form id="form">{% for section in sections %}<div class="formgrid {% if section != 'Profile' %}hidden{% endif %}" data-section="{{section}}">{% for f in features if f.section == section %}<div class="field"><label for="f_{{f.key}}">{{f.label}}</label>{% if f.kind == 'select' or f.kind == 'toggle2' %}<select id="f_{{f.key}}" name="{{f.key}}">{% for value,label in f.options %}<option value="{{value}}" {% if value == f.default %}selected{% endif %}>{{label}}</option>{% endfor %}</select>{% elif f.kind == 'toggle' %}<select id="f_{{f.key}}" name="{{f.key}}"><option value="1" {% if f.default == '1' %}selected{% endif %}>Yes</option><option value="0" {% if f.default == '0' %}selected{% endif %}>No</option></select>{% elif f.kind == 'range' %}<div class="range-head"><span style="font-size:11px;color:var(--muted)">Adjust value</span><output id="o_{{f.key}}">{{f.default}}</output></div><input type="range" id="f_{{f.key}}" name="{{f.key}}" min="{{f.min}}" max="{{f.max}}" step="{{f.step}}" value="{{f.default}}" oninput="document.getElementById('o_{{f.key}}').textContent=this.value">{% else %}<input type="number" id="f_{{f.key}}" name="{{f.key}}" min="{{f.min}}" max="{{f.max}}" step="{{f.step}}" value="{{f.default}}" required>{% endif %}<small>{{f.help}}</small></div>{% endfor %}</div>{% endfor %}<div class="actions"><button class="btn" type="reset" id="reset">Reset inputs</button><button class="btn primary" type="submit" id="run">Run prediction →</button></div><div class="alert" id="error" role="alert"></div></form></article><div class="result"><article class="card"><div class="heading"><h2>Churn probability</h2><span>ANN inference</span></div><div class="donut" id="donut"><div class="donut-inner"><strong id="prob">—</strong><small>Predicted risk</small></div></div><div class="risk" id="risk">Awaiting assessment</div><p class="note" style="text-align:center">This model output is an estimate, not a guaranteed outcome.</p></article><article class="card"><div class="heading"><h2>Scenario sensitivity</h2><span>One-feature changes</span></div><div id="drivers" class="note">Run a prediction to inspect how changing individual inputs affects the model output.</div></article></div></section><section class="bottom" id="analytics"><article class="card"><div class="heading"><h2>Probability distribution</h2><span>Recorded runs</span></div><div class="bars" id="bars"></div><div class="axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div></article><article class="card" id="history"><div class="heading"><h2>Recent assessments</h2><button class="btn" type="button" id="clear">Clear</button></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Risk probability</th><th>Category</th></tr></thead><tbody id="recent"><tr><td colspan="3">No predictions yet</td></tr></tbody></table></div></article></section><div class="foot">Demonstration: original fitted scaler and verified training preprocessing were not supplied; predictions may differ from the original model. Sensitivity comparisons are not causal explanations. Analytics are held in temporary server memory and may reset or vary between Vercel instances.</div></main></div>
<script>
const $=id=>document.getElementById(id);const features={{features|tojson}};const tabs=document.querySelectorAll('[data-tab]');tabs.forEach(b=>b.addEventListener('click',()=>{tabs.forEach(t=>t.classList.toggle('active',t===b));document.querySelectorAll('[data-section]').forEach(s=>s.classList.toggle('hidden',s.dataset.section!==b.dataset.tab))}));$('themeBtn').addEventListener('click',()=>{const html=document.documentElement;html.dataset.theme=html.dataset.theme==='dark'?'light':'dark'});$('form').addEventListener('submit',async e=>{e.preventDefault();$('error').style.display='none';const data={};for(const f of features){const n=Number($('f_'+f.key).value);if(!Number.isFinite(n)){$('error').textContent='Please enter valid numbers.';$('error').style.display='block';return}data[f.key]=n}const btn=$('run');btn.disabled=true;btn.textContent='Calculating…';try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const v=await r.json();if(!r.ok)throw Error(v.error||'Prediction failed');renderPrediction(v);renderStats(v.stats)}catch(err){$('error').textContent=err.message;$('error').style.display='block'}finally{btn.disabled=false;btn.textContent='Run prediction →'}});$('form').addEventListener('reset',()=>setTimeout(()=>features.filter(f=>f.kind==='range').forEach(f=>$('o_'+f.key).textContent=$('f_'+f.key).value),0));function renderPrediction(d){const p=d.probability;const color=p<30?'var(--secondary)':p<60?'var(--warning)':'var(--danger)';$('donut').style.setProperty('--pct',p);$('donut').style.setProperty('--ring',color);$('prob').textContent=p.toFixed(1)+'%';$('risk').textContent=d.risk+' risk';$('risk').style.color=color;const parent=$('drivers');parent.replaceChildren();for(const item of d.impacts.slice(0,5)){const row=document.createElement('div');row.className='driver';const top=document.createElement('div');top.className='driver-top';const name=document.createElement('span');name.textContent=item.label;const delta=document.createElement('span');delta.textContent=(item.delta>0?'+':'')+item.delta.toFixed(2)+' pp';top.append(name,delta);const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=item.pct+'%';fill.style.background=item.delta>0?'var(--danger)':'var(--secondary)';track.append(fill);row.append(top,track);parent.append(row)}}function renderStats(s){$('kTotal').textContent=s.total;$('kAvg').textContent=s.total?s.avg_prob+'%':'—';$('kHigh').textContent=s.high_risk;$('kChurn').textContent=s.total?s.churn_rate+'%':'—';const bars=$('bars');bars.replaceChildren();const max=Math.max(1,...s.buckets);s.buckets.forEach((n,i)=>{const b=document.createElement('div');b.className='bar';b.style.height=Math.max(3,n/max*140)+'px';b.title=`${i*10}–${(i+1)*10}%: ${n} runs`;bars.append(b)});const recent=$('recent');recent.replaceChildren();if(!s.recent.length){const tr=document.createElement('tr');const td=document.createElement('td');td.colSpan=3;td.textContent='No predictions yet';tr.append(td);recent.append(tr)}else s.recent.forEach(x=>{const tr=document.createElement('tr');for(const val of [x.ts,(x.probability*100).toFixed(1)+'%',x.risk]){const td=document.createElement('td');td.textContent=val;tr.append(td)}recent.append(tr)})}$('clear').addEventListener('click',async()=>{try{const r=await fetch('/api/reset',{method:'POST'});if(!r.ok)throw Error('Reset failed');renderStats(await r.json())}catch(e){alert(e.message)}});fetch('/api/stats').then(r=>r.json()).then(renderStats).catch(()=>{});
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
