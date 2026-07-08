# MLflow — Hands-On (Theory + Code)

MLflow is on the JD (MLOps tooling). Here's what it is and how to actually use it.

---

## 🧠 THEORY: what problem does MLflow solve?

When you build/tune models or LLM apps, you run **many experiments** — different prompts, models, temperatures, parameters. Without a system you lose track: *"Which settings gave the best result again?"*

**MLflow tracks every experiment** — the settings you used (parameters), the results you got (metrics), and the files produced (artifacts) — so you can compare runs and reproduce the best one. It's a lab notebook for ML/AI, plus a versioned model store.

---

## The 4 parts of MLflow (know these)
| Component | What it does |
|-----------|--------------|
| **Tracking** ⭐ | Log parameters, metrics, artifacts per run; compare in a UI | 
| **Models** | Package a model in a standard format so anyone can load/serve it |
| **Model Registry** | A versioned store of models with stages: Staging → Production |
| **Projects** | Package code so runs are reproducible |

For a fresher, **Tracking + Registry** are what you'll use and be asked about.

---

## 💻 CODE: your first tracked experiment

Install + run:
```powershell
uv run --with mlflow python mlflow_demo.py
```

`mlflow_demo.py`:
```python
import mlflow

mlflow.set_experiment("my-llm-experiment")   # groups related runs

with mlflow.start_run(run_name="prompt-v1"):
    # 1) log the settings you used (parameters — inputs you chose)
    mlflow.log_param("model", "gpt-4o-mini")
    mlflow.log_param("temperature", 0.2)
    mlflow.log_param("prompt_version", "v1")

    # 2) pretend you evaluated your app and got these results
    accuracy = 0.86
    avg_latency = 1.4

    # 3) log the results (metrics — numbers you measured)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("avg_latency_sec", avg_latency)

    # 4) log a file produced by the run (artifact)
    with open("sample_output.txt", "w") as f:
        f.write("The model answered correctly on 86% of test cases.")
    mlflow.log_artifact("sample_output.txt")

    print("Logged run to MLflow!")
```

**What just happened:** MLflow created a `mlruns/` folder and saved this run — its params, metrics, and the artifact file.

---

## 💻 See it in the UI

```powershell
uv run --with mlflow mlflow ui
```
Open **http://localhost:5000** → click your experiment → you see the run with all its params, metrics, and files. Run the script again with `temperature=0.7` and you get a **second run** you can compare side by side.

---

## 💻 CODE: comparing experiments (the real use)
```python
import mlflow

mlflow.set_experiment("prompt-tuning")

for temp in [0.0, 0.3, 0.7]:              # try 3 settings
    with mlflow.start_run(run_name=f"temp-{temp}"):
        mlflow.log_param("temperature", temp)
        # (imagine you ran your eval set and measured a score)
        score = 0.9 - abs(temp - 0.2)     # fake score, best near 0.2
        mlflow.log_metric("eval_score", score)
```
Now the UI lets you **sort runs by eval_score** and instantly see which temperature won. That's experiment tracking.

---

## 🧠 THEORY: the Model Registry (versioning)
When you have a good model, you **register** it. The registry keeps versions (v1, v2, v3) and lets you mark one as **Production** and another as **Staging**. If a new version underperforms, you **roll back** to the previous one. This is how teams manage which model is live.
```python
# register a logged model (conceptually)
mlflow.register_model("runs:/<run_id>/model", "my-model")
# then in the UI: promote a version to "Production" or roll back
```

---

## 🧠 How MLflow fits GenAI (say this in interviews)
> "I use MLflow Tracking to log every experiment — model, prompt version, temperature as parameters, and eval scores, latency, and token cost as metrics — so I can compare prompt/model changes objectively. Good models go into the Model Registry with staging/production stages, which gives me versioning and instant rollback if a new version degrades metrics."

---

## MLflow vs Weights & Biases (W&B) — common question
Both do experiment tracking + dashboards. **MLflow** is open-source, self-hostable, with a built-in model registry — common in enterprises. **W&B** is a polished hosted platform, strong for deep-learning training runs and sweeps. Same core idea; pick per team.

---

## ✏️ PRACTICE
1. Run `mlflow_demo.py`, then open the UI and find your run.
2. Run it again with different params (change temperature) — compare the two runs in the UI.
3. Run the "comparing experiments" loop and sort by `eval_score` in the UI.
4. Explain out loud: what's a *parameter* vs a *metric*? (Param = input you chose; metric = result you measured.)

---

## ✅ You now know
What MLflow is (experiment tracking + model registry), the 4 components, how to log params/metrics/artifacts, view runs in the UI, compare experiments, and version models with rollback — plus how to talk about it for GenAI.
