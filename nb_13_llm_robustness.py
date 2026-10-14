"""nb_13_llm_robustness - CSB Reports revision: everything that needs the LLM, with full run logging.

  A. Naive-LLM repeats (R1-8, R2, R3): N_RUNS_A runs at the ORIGINAL decoding (no temperature argument,
     as in nb_10b) + N_RUNS_T0 runs at temperature=0. Per-run labels are saved; accuracy mean/SD/CI and
     per-cluster label stability are reported.
  B. Single-LLM ablation (R1-2, R2): ONE model call sees ALL agent evidence as unstructured text and must
     return call + confidence + abstain. Same model as the framework. Isolates what the deterministic
     orchestrator adds beyond "just give the LLM the evidence".
  C. Narrator hallucination audit (R3): constrained narrator (nb_10b prompt) vs unconstrained prompt, SAME
     evidence, SAME model, N_RUNS_C runs per cluster. Rule-based checks (invented IDs, invented gene
     symbols) + an LLM claim-entailment judge + a CSV of random outputs for manual expert review.
  D. Run log: model id echoed by the API, SDK version, prompts, decoding, timestamps (R2 minor point).

Provider (default "anthropic" = Claude Haiku 4.5, the model of the reported runs; optional PROVIDER = "azure_openai" / "openai").
The API key is requested by a HIDDEN prompt when the script starts (or read from a notebook variable / environment
variable if you set one). It is never printed or saved.
Runner - TWO cells (recommended in Fabric; the hidden prompt inside exec() can hang):
  cell 1:  import getpass; ANTHROPIC_API_KEY = getpass.getpass("Claude key: ")
  cell 2:  exec(open("/lakehouse/default/Files/scRNA/code/nb_13_llm_robustness.py").read())
If cell 1 is skipped the script falls back to an environment variable, then to a hidden prompt.
Skip a part by setting its run count to 0 (N_RUNS_A / N_RUNS_B / N_RUNS_C = 0). Progress prints every 25 API calls;
results are checkpointed after parts A and B (llm_robustness_partial.json). Calls time out after 60 s.
Optional before running:  DATASETS = ["GSE131907"]   N_RUNS_A = 3   (quick test)   MODEL = "claude-haiku-4-5-20251001"
Writes: agent_evidence/<DS>/llm_robustness.json, ROOT/reanalysis/halluc_manual_review_<DS>.csv
"""
import json, re, time, datetime, random
print("nb_13 starting...", flush=True)
from pathlib import Path
import numpy as np, pandas as pd

DATASETS = globals().get("DATASETS", ["GSE131907", "GSE132465"])
N_RUNS_A = globals().get("N_RUNS_A", 10); N_RUNS_T0 = globals().get("N_RUNS_T0", 3)
N_RUNS_B = globals().get("N_RUNS_B", 5);  N_RUNS_C = globals().get("N_RUNS_C", 3)
import os, getpass
def _secret(name, prompt):
    print("(reading key: notebook variable -> environment -> hidden prompt)", flush=True)
    """Key from (1) notebook variable, (2) environment variable, (3) hidden prompt. Never printed, never written to disk."""
    v = globals().get(name) or os.environ.get(name)
    if not v: v = getpass.getpass(prompt)
    return v.strip()
PROVIDER = globals().get("PROVIDER", "anthropic")
if PROVIDER == "azure_openai":
    AZURE_ENDPOINT = globals().get("AZURE_ENDPOINT") or input("Azure endpoint: ").strip()
    AZURE_API_VERSION = globals().get("AZURE_API_VERSION") or input("Azure API version: ").strip()
    AZURE_DEPLOYMENT = globals().get("AZURE_DEPLOYMENT") or input("Azure deployment name: ").strip()
    AZURE_API_KEY = _secret("AZURE_API_KEY", "Azure OpenAI key (hidden): ")
    MODEL = globals().get("MODEL", AZURE_DEPLOYMENT)
elif PROVIDER == "openai":
    OPENAI_API_KEY = _secret("OPENAI_API_KEY", "OpenAI key (hidden): ")
    MODEL = globals().get("MODEL") or input("OpenAI model name: ").strip()
elif PROVIDER == "anthropic":
    ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY", "Claude API key sk-ant-... (hidden): ")
    assert ANTHROPIC_API_KEY.startswith("sk-ant-"), "that does not look like a Claude key (should start with sk-ant-)"
    MODEL = globals().get("MODEL", "claude-haiku-4-5-20251001")
else: raise ValueError(PROVIDER)
JUDGE_MODEL = globals().get("JUDGE_MODEL", MODEL)                # set a stronger model here if available
TISSUES = globals().get("TISSUES", {"GSE131907": "lung (lung adenocarcinoma)", "GSE132465": "colon (colorectal cancer)",
                                    "GSE176078": "breast (breast cancer)"})
ROOT = Path("/lakehouse/default/Files/scRNA"); (ROOT/"reanalysis").mkdir(exist_ok=True)

import subprocess, sys
def _imp(mod, pipname):
    try: return __import__(mod)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pipname, "--break-system-packages"]); return __import__(mod)
import requests
if PROVIDER == "anthropic":
    _client = None; _sdk = "direct HTTPS (requests " + requests.__version__ + "), anthropic-version 2023-06-01"   # no anthropic SDK needed
else:
    openai = _imp("openai", "openai")
    _client = (openai.AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, timeout=60.0, max_retries=2)
               if PROVIDER == "azure_openai" else openai.OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2))
    _sdk = "openai " + openai.__version__
print(f"client ready: provider={PROVIDER} model={MODEL}", flush=True)
CALL_LOG = {"provider": PROVIDER, "sdk_version": _sdk, "models_echoed": set(), "n_calls": 0, "temperature_unsupported": False,
            "api_version": globals().get("AZURE_API_VERSION"), "started": datetime.datetime.utcnow().isoformat()}
_REASONING = PROVIDER != "anthropic" and str(MODEL).lower().startswith(("gpt-5", "o1", "o3", "o4"))
_ST = {"tok_key": "max_completion_tokens", "reasoning_effort": _REASONING}   # adapts to what the deployment accepts

_T0 = time.time()
def _tick():
    CALL_LOG["n_calls"] += 1
    if CALL_LOG["n_calls"] % 25 == 0: print(f"    [{CALL_LOG['n_calls']} API calls, {time.time()-_T0:.0f}s elapsed]", flush=True)

def call(system, user, max_tokens=300, temperature=None, model=None, retries=6):
    """One completion. Returns '' after repeated failure (logged). temperature=None -> provider default decoding."""
    m = model or MODEL
    if PROVIDER == "anthropic":
        body = dict(model=m, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        if temperature is not None: body["temperature"] = temperature
        hdr = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        for a in range(retries):
            try:
                resp = requests.post("https://api.anthropic.com/v1/messages", headers=hdr, json=body, timeout=60)
                if resp.status_code == 200:
                    j = resp.json(); _tick(); CALL_LOG["models_echoed"].add(j.get("model", m))
                    return "".join(b_.get("text", "") for b_ in j.get("content", []) if b_.get("type") == "text").strip()
                if resp.status_code in (401, 403):
                    print("  AUTH ERROR", resp.status_code, resp.text[:150]); return ""
                if resp.status_code in (429, 500, 502, 503, 529):
                    time.sleep(float(resp.headers.get("retry-after", 3*(a+1)))); continue
                print("  LLM call failed:", resp.status_code, resp.text[:150]); return ""
            except Exception as e:
                if a < retries-1: time.sleep(3*(a+1)); continue
                print("  LLM call failed:", str(e)[:120]); return ""
        return ""
    for a in range(retries):
        kw = dict(model=m, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        kw[_ST["tok_key"]] = max_tokens*(8 if _REASONING else 1)       # reasoning models spend tokens before answering
        if temperature is not None and not CALL_LOG["temperature_unsupported"]: kw["temperature"] = temperature
        if _ST["reasoning_effort"]: kw["reasoning_effort"] = "low"
        try:
            r = _client.chat.completions.create(**kw); _tick(); CALL_LOG["models_echoed"].add(r.model)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e).lower()
            if "max_completion_tokens" in msg and _ST["tok_key"] == "max_completion_tokens" and "unsupported" in msg: _ST["tok_key"] = "max_tokens"; continue
            if "max_tokens" in msg and "max_completion_tokens" in msg and _ST["tok_key"] == "max_tokens": _ST["tok_key"] = "max_completion_tokens"; continue
            if "temperature" in msg and "temperature" in kw: CALL_LOG["temperature_unsupported"] = True; print("  note: this model ignores temperature; 'temp0' runs use default decoding"); continue
            if "reasoning_effort" in msg and _ST["reasoning_effort"]: _ST["reasoning_effort"] = False; continue
            if a < retries-1: time.sleep(3*(a+1)); continue
            print("  LLM call failed:", str(e)[:120]); return ""
    return ""

def jparse(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

# ---- same lineage map as nb_10a/nb_12 ----
_FAM = [("alveolar macrophage","myeloid"),("goblet","epithelial"),("ductal","epithelial"),("secretory","epithelial"),
 ("tuft","epithelial"),("enterocyte","epithelial"),("colonocyte","epithelial"),("paneth","epithelial"),
 ("enteroendocrine","epithelial"),("ciliated","epithelial"),("alveolar","epithelial"),("club","epithelial"),
 ("surfactant","epithelial"),("luminal","epithelial"),("basal cell","epithelial"),("lung cell","epithelial"),
 ("lung epithel","epithelial"),("cancer","epithelial"),("malignant","epithelial"),("carcinoma","epithelial"),
 ("epitheli","epithelial"),("mast","myeloid"),("plasmacytoid dendritic","myeloid"),("plasmacytoid","myeloid"),
 ("dendritic","myeloid"),("macrophage","myeloid"),("classical monocyte","myeloid"),("monocyte","myeloid"),
 ("neutrophil","myeloid"),("microglia","myeloid"),("myeloid","myeloid"),("fibroblast","stromal"),
 ("myofibroblast","stromal"),("endotheli","stromal"),("mesenchym","stromal"),("pericyte","stromal"),
 ("smooth muscle","stromal"),("emt","stromal"),("stromal","stromal"),("t/nk","lymphoid"),("nk/t","lymphoid"),
 ("natural killer","lymphoid"),("nkt","lymphoid"),("nk cell","lymphoid"),("plasma cell","lymphoid"),
 ("plasma","lymphoid"),("cd4","lymphoid"),("cd8","lymphoid"),("cytotoxic t","lymphoid"),("regulatory t","lymphoid"),
 ("treg","lymphoid"),("mature b","lymphoid"),("b cell","lymphoid"),("b lymph","lymphoid"),("t lymph","lymphoid"),
 ("t cell","lymphoid"),("lymphocyte","lymphoid"),("t-cell","lymphoid"),("b-cell","lymphoid"),("caf","stromal"),("pvl","stromal"),("perivascular","stromal")]
def family(label):
    t = (label or "").lower()
    for k, f in _FAM:
        if k in t: return f
    return None

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(c-h), float(c+h)]
def auc(y, c):
    y = np.asarray(y); c = np.asarray(c, float); pos = c[y == 1]; neg = c[y == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5*np.sum(pos[:, None] == neg[None, :]))/(len(pos)*len(neg)))

NARR_SYS = (  # IDENTICAL to nb_10b
    "You write the functional interpretation of a single-cell cluster for a scientific paper, "
    "STRICTLY from the structured evidence provided. Hard rules:\n"
    "1. Cite ONLY pathway/disease names that appear VERBATIM in the evidence JSON. Never introduce "
    "a pathway, gene, disease, or database ID that is not in the input.\n"
    "2. If the top pathways are generic/housekeeping (translation, ribosomal, amino-acid sensing, "
    "EIF2AK4/GCN2, peptide chain elongation), SAY the pathway evidence is non-specific for this "
    "cluster and do NOT spin a biological story from them.\n"
    "3. Do NOT invent mechanistic explanations, marker genes, or disease IDs from your own knowledge.\n"
    "4. If the cell-type call conflicts with the pathway lineage or is flagged, report the "
    "disagreement plainly and note the call is uncertain.\n"
    "5. Write 2-3 factual sentences: (a) the call + confidence, (b) which SPECIFIC provided "
    "pathways/disease hits support or contradict it (or state none are specific), (c) any flagged "
    "uncertainty. No embellishment.")
UNCON_SYS = ("You write the functional interpretation of a single-cell cluster for a scientific paper. "
             "Write 2-3 sentences interpreting the cluster using the evidence provided.")
JUDGE_SYS = ("You are a strict fact-checker. Given EVIDENCE (JSON) and a TEXT, split the TEXT into atomic factual claims. "
             "A claim is SUPPORTED only if it is stated in, or directly entailed by, the EVIDENCE. Claims that rely on outside "
             "knowledge (extra genes, mechanisms, disease IDs, pathways not listed) are UNSUPPORTED. Hedges about uncertainty "
             "are not claims. Reply ONLY with JSON: {\"n_claims\": int, \"n_unsupported\": int, \"unsupported\": [short strings]}")
ID_RE = re.compile(r"(R-HSA-\d+|GO:\d{5,7}|MONDO[_:]\d+|EFO[_:]\d+|HP[_:]\d+|rs\d{3,}|ENSG\d+)")
GENE_RE = re.compile(r"\b[A-Z][A-Z0-9-]{2,9}\b")
ALLOW = set("""DNA RNA FDR GO NK NKT CD4 CD8 EMT TAM DEG DEGS GWAS ORA UMAP LUAD CRC HPA TNF IL TGF NF MHC TCR BCR GSEA
 MSIGDB REACTOME HALLMARK DISEASE ABSTAIN REVIEW AGENT MARKER PATHWAY BP T B TCR TLR NKT CAF APC GCN2 BCR NKG""".split())

for DATASET in DATASETS:
    print("\n" + "#"*72 + f"\n# {DATASET}\n" + "#"*72)
    DEG = ROOT/"DEG"/DATASET; EVID = ROOT/"agent_evidence"/DATASET; TISSUE = TISSUES[DATASET]
    gt = pd.read_csv(DEG/"cluster_ground_truth.csv", index_col=0); gt.index = gt.index.astype(str)
    tops = json.load(open(DEG/"degs_top.json")); CL = list(tops.keys())
    orch = json.load(open(EVID/"orchestrated.json")); pathway = json.load(open(EVID/"pathway_evidence.json"))
    disease = json.load(open(EVID/"disease_evidence.json")); marker = json.load(open(EVID/"marker_expression_evidence.json"))
    afam = {cl: family(gt.loc[cl, "majority_author_label"]) for cl in CL if cl in gt.index}
    R = dict(dataset=DATASET, model=MODEL, judge_model=JUDGE_MODEL, N=dict(A=N_RUNS_A, T0=N_RUNS_T0, B=N_RUNS_B, C=N_RUNS_C))

    # ---------------- A. naive repeats ----------------
    naive_sys = (f"You are a single-cell RNA-seq expert. Given a cluster's top marker genes from human {TISSUE} tissue, "
                 f"name the MOST LIKELY single cell type. Reply with ONLY the cell type name, nothing else.")
    runsA = {"default": [], "temp0": []}
    for cfg, nrun, temp in [("default", N_RUNS_A, None), ("temp0", N_RUNS_T0, 0.0)]:
        for r in range(nrun):
            lab = {cl: call(naive_sys, f"Top marker genes: {', '.join(tops[cl]['top_agents'][:20])}\nCell type:", 30, temp) for cl in CL}
            runsA[cfg].append(lab)
            acc = np.mean([family(lab[cl]) == afam[cl] for cl in afam])
            print(f"  A {cfg:7} run {r+1}: acc={acc:.3f}")
    A = {}
    for cfg, runs in runsA.items():
        accs = [float(np.mean([family(l[cl]) == afam[cl] for cl in afam])) for l in runs]
        stab = {cl: float(max([sum(family(l[cl]) == f for l in runs) for f in set(family(l[cl]) for l in runs)])/len(runs)) for cl in afam} if runs else {}
        A[cfg] = dict(run_accuracies=accs, mean=float(np.mean(accs)) if accs else None,
                      sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                      min=min(accs) if accs else None, max=max(accs) if accs else None,
                      pooled_calls_ci=wilson(int(round(np.mean(accs)*len(afam)*len(accs))), len(afam)*len(accs)) if accs else None,
                      clusters_with_unstable_family=[cl for cl, s in stab.items() if s < 1.0], labels=runs)
        if accs: print(f"  A {cfg}: mean={A[cfg]['mean']:.3f} sd={A[cfg]['sd']} range=[{A[cfg]['min']:.3f},{A[cfg]['max']:.3f}] unstable clusters={A[cfg]['clusters_with_unstable_family']}")
    R["A_naive_repeats"] = A
    json.dump(R, open(EVID/"llm_robustness_partial.json", "w"), indent=1, default=str); print("  checkpoint A saved", flush=True)

    # ---------------- B. single-LLM with all evidence ----------------
    def evidence_text(cl):
        m = marker[cl]; p = pathway.get(cl, {}); d = disease.get(cl, {})
        return json.dumps(dict(top_DEGs=tops[cl]["top_agents"][:20],
                               marker_candidates=m.get("candidates"),
                               pathway_terms=[r["object"] for r in p.get("records", [])[:10]],
                               disease_hits=[r.get("object") for r in d.get("records", [])[:10]],
                               driver_genes=d.get("drivers", [])), indent=1)
    sys_B = (f"You are a single-cell expert. For human {TISSUE} tissue, use ALL the evidence below to name the cluster's most likely cell type, "
             "give a confidence in [0,1] reflecting how reliable the call is, and set abstain=true if the evidence is too weak or contradictory. "
             "Reply ONLY with JSON: {\"cell_type\": str, \"confidence\": float, \"abstain\": bool}")
    runsB = []
    for r in range(N_RUNS_B):
        out = {}
        for cl in CL:
            j = jparse(call(sys_B, f"Cluster {cl} evidence:\n{evidence_text(cl)}", 120))
            out[cl] = dict(cell_type=j.get("cell_type"), confidence=j.get("confidence"), abstain=j.get("abstain"))
        runsB.append(out)
    accB, aucB = [], []
    for out in runsB:
        ok = [int(family(out[cl]["cell_type"]) == afam[cl]) for cl in afam]
        cf = [float(out[cl]["confidence"]) if isinstance(out[cl]["confidence"], (int, float)) else 0.0 for cl in afam]
        accB.append(float(np.mean(ok))); aucB.append(auc(ok, cf))
    fw_ok = [int(family(orch[cl]["marker_call"]) == afam[cl]) for cl in afam]
    R["B_single_llm_with_evidence"] = dict(run_accuracy=accB, mean_acc=float(np.mean(accB)) if accB else None,
        run_auroc_selfreported_conf=aucB, framework_accuracy=float(np.mean(fw_ok)),
        framework_auroc=auc(fw_ok, [orch[cl]["confidence"] for cl in afam]),
        mean_abstain_rate=float(np.mean([np.mean([bool(o[cl]["abstain"]) for cl in CL]) for o in runsB])) if runsB else None,
        framework_abstain_rate=float(np.mean([orch[cl]["abstain"] for cl in CL])), runs=runsB)
    print(f"  B single-LLM+evidence: acc mean={np.mean(accB):.3f} runs={['%.2f' % a for a in accB]}  AUROC(self-conf)={aucB}  | framework acc={np.mean(fw_ok):.3f}")

    json.dump(R, open(EVID/"llm_robustness_partial.json", "w"), indent=1, default=str); print("  checkpoint B saved", flush=True)

    # ---------------- C. narrator hallucination audit ----------------
    def narr_evidence(cl):
        o = orch[cl]
        return dict(cell_type_call=o.get("marker_call"), confidence=o.get("confidence"), abstained=o.get("abstain"),
                    cross_agent_flags=o.get("flags"),
                    top_pathways=[r["object"] for r in pathway.get(cl, {}).get("records", [])[:4]],
                    disease_evidence_objects=[r.get("object") for r in disease.get(cl, {}).get("records", [])[:6]
                                              if r.get("details", {}).get("context_relevant")],
                    disease_context_hit_count=disease.get(cl, {}).get("n_context_relevant", 0),
                    note="If top_pathways are translation/ribosomal/EIF2AK4 terms, they are NON-SPECIFIC.")
    C = {}; manual = []
    for cond, sysmsg in [("constrained", NARR_SYS), ("unconstrained", UNCON_SYS)]:
        tot_claims = tot_unsup = tot_idbad = tot_genebad = n_texts = 0; per = []
        for cl in CL:
            ev = narr_evidence(cl); evs = json.dumps(ev, indent=1); evtext = evs.upper()
            for r in range(N_RUNS_C):
                txt = call(sysmsg, f"Cluster {cl} evidence (JSON):\n{evs}\n\nWrite the interpretation:", 250)
                idbad = [i for i in ID_RE.findall(txt) if i.upper() not in evtext]
                genebad = [g for g in (x.rstrip("-") for x in GENE_RE.findall(txt)) if g and g not in ALLOW and g not in evtext and not g.isdigit()]
                jj = jparse(call(JUDGE_SYS, f"EVIDENCE:\n{evs}\n\nTEXT:\n{txt}", 400, 0.0, JUDGE_MODEL))
                nc = int(jj.get("n_claims", 0) or 0); nu = int(jj.get("n_unsupported", 0) or 0)
                tot_claims += nc; tot_unsup += nu; tot_idbad += len(idbad); tot_genebad += len(genebad); n_texts += 1
                per.append(dict(cluster=cl, run=r, n_claims=nc, n_unsupported=nu, bad_ids=idbad, bad_genes=genebad,
                                unsupported=jj.get("unsupported", [])))
                manual.append(dict(dataset=DATASET, condition=cond, cluster=cl, run=r, evidence=evs, text=txt,
                                   judge_unsupported=json.dumps(jj.get("unsupported", [])), expert_n_unsupported=""))
                time.sleep(0.2)
        C[cond] = dict(n_texts=n_texts, claims=tot_claims, unsupported=tot_unsup,
                       unsupported_rate=(tot_unsup/tot_claims if tot_claims else None),
                       texts_with_invented_id=sum(1 for p in per if p["bad_ids"]),
                       texts_with_invented_gene=sum(1 for p in per if p["bad_genes"]),
                       texts_with_any_unsupported_claim=sum(1 for p in per if p["n_unsupported"] > 0), detail=per)
        print(f"  C {cond:13}: {n_texts} texts, claims={tot_claims}, judge-unsupported={tot_unsup} "
              f"({C[cond]['unsupported_rate']}), texts w/ invented ID={C[cond]['texts_with_invented_id']}, "
              f"invented gene symbol={C[cond]['texts_with_invented_gene']}, any unsupported={C[cond]['texts_with_any_unsupported_claim']}")
    R["C_hallucination_audit"] = C
    random.seed(0); random.shuffle(manual)
    pd.DataFrame(manual[:max(40, len(manual)//5)]).to_csv(ROOT/"reanalysis"/f"halluc_manual_review_{DATASET}.csv", index=False)
    print(f"  manual-review sample -> reanalysis/halluc_manual_review_{DATASET}.csv (two experts fill expert_n_unsupported; compute kappa)")

    CALL_LOG["models_echoed"] = sorted(CALL_LOG["models_echoed"])
    R["D_run_log"] = dict(CALL_LOG); R["prompts"] = dict(naive=naive_sys, single_llm=sys_B, narrator_constrained=NARR_SYS,
                                                        narrator_unconstrained=UNCON_SYS, judge=JUDGE_SYS)
    R["D_run_log"]["finished"] = datetime.datetime.utcnow().isoformat()
    json.dump(R, open(EVID/"llm_robustness.json", "w"), indent=1, default=str)
    CALL_LOG["models_echoed"] = set(CALL_LOG["models_echoed"])
    print(f"  saved {EVID/'llm_robustness.json'}")

print("\nNOTEBOOK 13 COMPLETE. Judge model:", JUDGE_MODEL, "- state it, and the manual-review kappa, in Methods.")
