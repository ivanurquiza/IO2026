"""
Diagnostico: sensibilidad de la estimacion a TAU, la dispersion supuesta
del ingreso dentro de cada tienda. TAU no se observa: lo fijamos en 0.5.
Este script reestima con TAU en {0.2, 0.5, 1.0} para ver cuanto dependen
de ese supuesto el coeficiente de precio y los sigma.

Correr:  python code/09_sensibilidad_tau.py
"""
import numpy as np, pandas as pd, pyblp
from pathlib import Path

pyblp.options.verbose = False
RAIZ = Path(__file__).resolve().parent.parent
prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
base = pd.read_csv(RAIZ / "data" / "base_limpia.csv")

SEM, NS, K2 = 2026, 200, 5     # NS=200 para que sea rapido; el final usa 500
integ = pyblp.build_integration(pyblp.Integration("halton", NS, {"seed": SEM}), K2 + 1)
nod, w = integ.nodes, integ.weights.flatten()
nv, ning = nod[:, :K2], nod[:, K2]
loging = base.groupby("tienda")["ingreso"].first()
merc = base[["mercado", "tienda"]].drop_duplicates().reset_index(drop=True)
T = len(merc); ms = loging.loc[merc["tienda"]].to_numpy()

X1 = pyblp.Formulation("0 + prices + promo + C(product_ids)")
X2 = pyblp.Formulation("0 + marca1 + marca2 + marca3 + marca4 + prices")
AG = pyblp.Formulation("0 + income_c")
s0 = np.diag([0.5, 0.5, 0.5, 0.5, 0.0])
p0 = np.array([[0.], [0.], [0.], [0.], [-0.05]])

for TAU in [0.2, 0.5, 1.0]:
    a = pd.DataFrame({"market_ids": np.repeat(merc["mercado"].values, NS),
                      "weights": np.tile(w, T)})
    for k in range(K2):
        a[f"nodes{k}"] = np.tile(nv[:, k], T)
    inc = (np.exp(ms[:, None] + TAU * ning[None, :]) / 10000.0).ravel()
    a["income_c"] = inc - inc.mean()
    r = pyblp.Problem((X1, X2), prod, AG, a).solve(
        sigma=s0, pi=p0, method="1s", se_type="robust",
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-4}))
    s = np.diag(r.sigma)
    print("TAU=%.1f | alpha=%.4f | sigmaB=%s | pi=%.4f | GMM=%.2f"
          % (TAU, r.beta[0, 0], np.round(s[:4], 3), r.pi[4, 0], r.objective.item()))