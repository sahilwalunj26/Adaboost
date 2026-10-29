"""
Admission Predictor — Flask app around adaboost.pkl (AdaBoostClassifier, 18 features).

Run:   python app.py        ->  http://127.0.0.1:5000
Prod:  gunicorn app:app
Keep adaboost.pkl in the same folder as this file.
"""
import os
import warnings

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaboost.pkl")

app = Flask(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

# NOTE: the model has no stored encoders. `gender` and `state` are assumed to be
# label-encoded (gender: 0/1/2, state: alphabetical index 0-49). This model gives
# both a feature importance of 0, so the choice does not change predictions.
STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming",
]

# name, label, group, kind, min, max, step, default, hint
FIELDS = [
    dict(name="age", label="Age", group="About the applicant", kind="slider", min=14, max=30, step=1, default=18, unit="yrs"),
    dict(name="gender", label="Gender", group="About the applicant", kind="select", default=0,
         options=[{"v": 0, "t": "Female"}, {"v": 1, "t": "Male"}, {"v": 2, "t": "Other"}]),
    dict(name="state", label="State", group="About the applicant", kind="select", default=4,
         options=[{"v": i, "t": s} for i, s in enumerate(STATES)]),
    dict(name="family_income", label="Family income", group="About the applicant", kind="number", min=0, max=1000000, step=1000, default=75000, unit="USD / year"),

    dict(name="high_school_gpa", label="High school GPA", group="Academics", kind="slider", min=0, max=4, step=0.01, default=3.5, unit="/ 4.0", key=True),
    dict(name="sat_score", label="SAT score", group="Academics", kind="slider", min=400, max=1600, step=10, default=1200, unit="/ 1600", key=True),
    dict(name="act_score", label="ACT score", group="Academics", kind="slider", min=1, max=36, step=1, default=26, unit="/ 36"),
    dict(name="attendance_rate", label="Attendance rate", group="Academics", kind="slider", min=50, max=100, step=1, default=95, unit="%"),
    dict(name="ap_courses", label="AP courses taken", group="Academics", kind="slider", min=0, max=15, step=1, default=4, unit=""),

    dict(name="extracurricular_count", label="Extracurriculars", group="Beyond the classroom", kind="slider", min=0, max=15, step=1, default=3, unit="activities"),
    dict(name="volunteer_hours", label="Volunteer hours", group="Beyond the classroom", kind="slider", min=0, max=500, step=5, default=60, unit="hrs"),
    dict(name="leadership_positions", label="Leadership positions", group="Beyond the classroom", kind="slider", min=0, max=10, step=1, default=1, unit=""),
    dict(name="coding_projects", label="Coding projects", group="Beyond the classroom", kind="slider", min=0, max=20, step=1, default=2, unit=""),
    dict(name="social_media_hours", label="Social media per day", group="Beyond the classroom", kind="slider", min=0, max=12, step=0.5, default=3, unit="hrs"),
    dict(name="online_certifications", label="Online certifications", group="Beyond the classroom", kind="slider", min=0, max=20, step=1, default=2, unit=""),

    dict(name="essay_score", label="Essay score", group="Application review", kind="slider", min=0, max=100, step=1, default=75, unit="/ 100", key=True),
    dict(name="recommendation_score", label="Recommendation score", group="Application review", kind="slider", min=0, max=100, step=1, default=78, unit="/ 100"),
    dict(name="interview_score", label="Interview score", group="Application review", kind="slider", min=0, max=100, step=1, default=72, unit="/ 100", key=True),
]

assert [f["name"] for f in FIELDS] == FEATURES, "FIELDS must match the model's feature order"

IMPORTANCE = {
    f: round(float(i), 4) for f, i in zip(FEATURES, model.feature_importances_)
}


def parse_payload(data):
    row = {}
    for f in FIELDS:
        raw = data.get(f["name"])
        if raw is None or raw == "":
            raise ValueError(f"{f['label']} is required.")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{f['label']} must be a number.")
        if f["kind"] != "select" and not (f["min"] <= val <= f["max"]):
            raise ValueError(f"{f['label']} must be between {f['min']} and {f['max']}.")
        row[f["name"]] = val
    return row


@app.post("/api/predict")
def predict():
    try:
        row = parse_payload(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400

    X = pd.DataFrame([row], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p_yes = float(proba[classes.index(1)])
    label = int(model.predict(X)[0])

    if p_yes >= 0.65:
        verdict = "Strong chance"
    elif p_yes >= 0.5:
        verdict = "Leaning yes"
    elif p_yes >= 0.35:
        verdict = "Leaning no"
    else:
        verdict = "Unlikely"

    return jsonify(prediction=label, probability=round(p_yes, 4), verdict=verdict)


@app.get("/health")
def health():
    return jsonify(status="ok", model=type(model).__name__, features=len(FEATURES))


@app.get("/")
def index():
    import json
    html = PAGE.replace("__FIELDS__", json.dumps(FIELDS)).replace("__IMPORTANCE__", json.dumps(IMPORTANCE))
    return Response(html, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admission Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --chalk:#EEF0F4; --paper:#FAFBFD; --ink:#111B33; --ink2:#1D2B4D; --mute:#5C6785;
  --line:#D6DBE6; --mark:#FFD23F; --mint:#3FBF9B; --rose:#E4576E;
  --serif:"Instrument Serif",Georgia,"Times New Roman",serif;
  --sans:"Manrope",system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box;margin:0}
html{scroll-behavior:smooth}
body{font-family:var(--sans);background:var(--chalk);color:var(--ink);min-height:100vh;line-height:1.5}
.shell{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:0;max-width:1280px;margin:0 auto}
/* ---------- form side ---------- */
main{padding:56px 48px 80px}
header h1{font-family:var(--serif);font-weight:400;font-size:clamp(40px,5vw,64px);line-height:1.02;letter-spacing:-.01em;max-width:14ch}
header p{margin-top:14px;color:var(--mute);max-width:52ch}
.group{margin-top:44px}
.group h2{font-family:var(--serif);font-weight:400;font-size:28px;padding-bottom:10px;border-bottom:2px solid var(--ink)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:26px 36px;margin-top:22px}
.field label{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;font-weight:600;margin-bottom:8px}
.field label .val{font-variant-numeric:tabular-nums;font-weight:700;background:var(--mark);padding:1px 8px;border-radius:4px;color:var(--ink)}
.field label .val small{font-weight:500;color:#5a4a00;margin-left:3px}
.field.key label::after{content:"";}
.tag{font-size:11px;font-weight:600;color:var(--mute);margin-left:6px;border:1px solid var(--line);border-radius:99px;padding:0 7px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:99px;outline:none;cursor:pointer;
  background:linear-gradient(to right,var(--ink) var(--p,50%),var(--line) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:var(--paper);border:3px solid var(--ink);transition:transform .12s}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--paper);border:3px solid var(--ink)}
input[type=range]:active::-webkit-slider-thumb{transform:scale(1.2)}
input[type=range]:focus-visible,select:focus-visible,input[type=number]:focus-visible,button:focus-visible{outline:3px solid var(--mark);outline-offset:3px}
select,input[type=number]{width:100%;padding:10px 12px;font:inherit;font-size:15px;background:var(--paper);border:1.5px solid var(--line);border-radius:8px;color:var(--ink)}
.bounds{display:flex;justify-content:space-between;font-size:11px;color:var(--mute);margin-top:5px}
/* ---------- verdict side ---------- */
aside{position:sticky;top:0;height:100vh;padding:40px 36px;background:var(--ink);color:#EAEEF8;display:flex;flex-direction:column}
aside h3{font-family:var(--serif);font-weight:400;font-size:22px;color:#AAB4D0}
.gauge{margin-top:18px;position:relative}
.gauge svg{width:100%;display:block}
.gauge .num{position:absolute;left:0;right:0;bottom:6px;text-align:center}
.num b{font-family:var(--serif);font-weight:400;font-size:84px;line-height:1;letter-spacing:-.02em}
.num span{font-family:var(--serif);font-size:32px;color:#AAB4D0}
.verdict{text-align:center;margin-top:10px;font-family:var(--serif);font-size:34px;min-height:42px}
.verdict.good{color:var(--mint)} .verdict.bad{color:var(--rose)} .verdict.mid{color:var(--mark)}
.sub{text-align:center;color:#AAB4D0;font-size:14px;min-height:22px}
button#go{margin-top:22px;width:100%;padding:15px;font:inherit;font-weight:700;font-size:16px;border:0;border-radius:10px;background:var(--mark);color:var(--ink);cursor:pointer;transition:transform .12s,filter .12s}
button#go:hover{filter:brightness(1.06)} button#go:active{transform:translateY(1px)}
button#go[disabled]{opacity:.6;cursor:progress}
.reset{margin-top:10px;background:none;border:0;color:#AAB4D0;font:inherit;font-size:13px;cursor:pointer;text-decoration:underline}
.weigh{margin-top:auto;padding-top:24px;border-top:1px solid #2C3A60}
.weigh h4{font-size:13px;font-weight:600;color:#AAB4D0;margin-bottom:10px}
.bar{display:grid;grid-template-columns:130px 1fr 38px;align-items:center;gap:10px;font-size:12.5px;margin:6px 0}
.bar i{display:block;height:7px;border-radius:99px;background:#2C3A60;overflow:hidden}
.bar i em{display:block;height:100%;background:var(--mark);border-radius:99px}
.bar span:last-child{text-align:right;color:#AAB4D0;font-variant-numeric:tabular-nums}
.err{color:#FFB3C0;text-align:center;font-size:14px;min-height:20px;margin-top:8px}
.arc-fill{transition:stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1),stroke .4s}
@media (max-width:980px){
  .shell{grid-template-columns:1fr}
  aside{position:static;height:auto;order:-1;padding:28px 24px}
  .weigh{margin-top:28px}
  main{padding:36px 22px 60px}
}
@media (max-width:560px){.grid{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="shell">
  <main>
    <header>
      <h1>Will this application get in?</h1>
      <p>Enter the applicant's profile. An AdaBoost model of ten decision stumps estimates the chance of admission.</p>
    </header>
    <form id="form" autocomplete="off"></form>
  </main>

  <aside>
    <h3>Admission chance</h3>
    <div class="gauge">
      <svg viewBox="0 0 300 165" aria-hidden="true">
        <path d="M 20 150 A 130 130 0 0 1 280 150" fill="none" stroke="#2C3A60" stroke-width="16" stroke-linecap="round"/>
        <path id="arc" class="arc-fill" d="M 20 150 A 130 130 0 0 1 280 150" fill="none" stroke="#FFD23F" stroke-width="16" stroke-linecap="round" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/>
      </svg>
      <div class="num" aria-live="polite"><b id="pct">--</b><span>%</span></div>
    </div>
    <div class="verdict" id="verdict">Ready</div>
    <div class="sub" id="sub">Adjust the profile, then run the prediction.</div>
    <button id="go" type="button">Predict admission</button>
    <button class="reset" id="reset" type="button">Reset to defaults</button>
    <div class="err" id="err" role="alert"></div>

    <div class="weigh">
      <h4>What this model weighs most</h4>
      <div id="bars"></div>
    </div>
  </aside>
</div>

<script>
const FIELDS = __FIELDS__;
const IMPORTANCE = __IMPORTANCE__;
const form = document.getElementById('form');

const fmt = (f, v) => f.name === 'family_income' ? Number(v).toLocaleString() : (f.step < 1 ? Number(v).toFixed(f.step < 0.1 ? 2 : 1) : v);

function build(){
  let group = null, grid = null;
  FIELDS.forEach(f => {
    if (f.group !== group){
      group = f.group;
      const sec = document.createElement('section'); sec.className = 'group';
      sec.innerHTML = `<h2>${group}</h2><div class="grid"></div>`;
      form.appendChild(sec); grid = sec.querySelector('.grid');
    }
    const w = document.createElement('div'); w.className = 'field' + (f.key ? ' key' : '');
    const tag = IMPORTANCE[f.name] > 0.05 ? '<span class="tag">counts</span>' : '';
    if (f.kind === 'slider'){
      w.innerHTML = `<label for="${f.name}"><span>${f.label}${tag}</span><span class="val"><span id="v_${f.name}">${fmt(f,f.default)}</span><small>${f.unit}</small></span></label>
        <input type="range" id="${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">
        <div class="bounds"><span>${f.min}</span><span>${f.max}</span></div>`;
    } else if (f.kind === 'select'){
      w.innerHTML = `<label for="${f.name}"><span>${f.label}</span></label>
        <select id="${f.name}" name="${f.name}">${f.options.map(o=>`<option value="${o.v}" ${o.v===f.default?'selected':''}>${o.t}</option>`).join('')}</select>`;
    } else {
      w.innerHTML = `<label for="${f.name}"><span>${f.label}</span><span class="val"><small style="margin:0">${f.unit}</small></span></label>
        <input type="number" id="${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">`;
    }
    grid.appendChild(w);
  });
  document.querySelectorAll('input[type=range]').forEach(r => {
    const f = FIELDS.find(x => x.name === r.id);
    const paint = () => {
      r.style.setProperty('--p', ((r.value - r.min) / (r.max - r.min) * 100) + '%');
      document.getElementById('v_' + r.id).textContent = fmt(f, r.value);
    };
    r.addEventListener('input', paint); paint();
  });
}

function buildBars(){
  const top = Object.entries(IMPORTANCE).filter(([,v]) => v > 0).sort((a,b) => b[1]-a[1]).slice(0,5);
  const max = top[0][1];
  document.getElementById('bars').innerHTML = top.map(([k,v]) => {
    const f = FIELDS.find(x => x.name === k);
    return `<div class="bar"><span>${f.label}</span><i><em style="width:${v/max*100}%"></em></i><span>${Math.round(v*100)}%</span></div>`;
  }).join('');
}

const arc = document.getElementById('arc'), pct = document.getElementById('pct'),
      verdict = document.getElementById('verdict'), sub = document.getElementById('sub'),
      err = document.getElementById('err'), go = document.getElementById('go');

function show(p, label, pred){
  const v = Math.round(p * 100);
  arc.style.strokeDashoffset = 100 - v;
  const cls = p >= 0.5 ? (p >= 0.65 ? 'good' : 'mid') : (p >= 0.35 ? 'mid' : 'bad');
  arc.style.stroke = cls === 'good' ? '#3FBF9B' : cls === 'bad' ? '#E4576E' : '#FFD23F';
  verdict.className = 'verdict ' + cls; verdict.textContent = label;
  sub.textContent = pred === 1 ? 'The model predicts: admitted.' : 'The model predicts: not admitted.';
  let cur = parseInt(pct.textContent) || 0, step = v > cur ? 1 : -1;
  const t = setInterval(() => { cur += step; pct.textContent = cur; if (cur === v) clearInterval(t); }, 900 / Math.max(1, Math.abs(v - cur)));
  if (cur === v) pct.textContent = v;
}

go.addEventListener('click', async () => {
  err.textContent = ''; go.disabled = true; go.textContent = 'Predicting...';
  const data = {};
  FIELDS.forEach(f => data[f.name] = document.getElementById(f.name).value);
  try {
    const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Prediction failed.');
    show(j.probability, j.verdict, j.prediction);
  } catch (e) { err.textContent = e.message; }
  go.disabled = false; go.textContent = 'Predict admission';
});

document.getElementById('reset').addEventListener('click', () => {
  FIELDS.forEach(f => { const el = document.getElementById(f.name); el.value = f.default; el.dispatchEvent(new Event('input')); });
  arc.style.strokeDashoffset = 100; pct.textContent = '--'; verdict.className = 'verdict'; verdict.textContent = 'Ready';
  sub.textContent = 'Adjust the profile, then run the prediction.'; err.textContent = '';
});

build(); buildBars();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)