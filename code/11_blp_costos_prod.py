"""
Script 11: Ejercicio 4c. Costos marginales bajo Bertrand MONOPRODUCTO

Bajo competencia de Bertrand-Nash entre firmas multiproducto, la
condicion de primer orden de cada mercado es (Conlon y Gortmaker 2020,
ec. 5):
        p = c + Delta(p,H)^{-1} s(p)
donde H es la matriz de propiedad: cada marca madre es dueña de sus
presentaciones y las fija conjuntamente. El costo marginal se recupera
como c = p - markup. PyBLP arma H a partir de firm_ids -que en nuestros
datos son las 4 marcas madre- y resuelve el sistema con compute_costs().

Se recuperan los costos con la demanda BLP y con la del logit, y se
comparan contra el costo observado en la base (el precio mayorista), en
la tienda 9 semana 10.

Correr:  python code/11_blp_costos_prod.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

pyblp.options.verbose = False
RAIZ = Path(__file__).resolve().parent.parent
MERCADO = "9-10"

with open(RAIZ / "output" / "blp_medios.pkl", "rb") as f:
    res_blp = pickle.load(f)

# demanda logit: mismo problema con dispersion nula
res_logit = res_blp.problem.solve(
    sigma=np.zeros((5, 5)), pi=np.zeros((5, 1)),
    method="1s", optimization=pyblp.Optimization("return"))

# ---------- costos marginales, cada producto su propio dueño -----------
# La consigna supone "cada marca tiene un solo dueno": cada producto se
# fija como monoproducto. Para que compute_costs use la matriz de
# propiedad identidad, pasamos un firm_ids con un dueno distinto por
# producto, recortado al mercado que se esta calculando.
prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
sub = prod[prod["market_ids"] == MERCADO].copy()
firm_mono = sub["product_ids"].to_numpy()       # 11 valores, uno por producto

costo_blp = res_blp.compute_costs(
    market_id=MERCADO, firm_ids=firm_mono).flatten()
costo_logit = res_logit.compute_costs(
    market_id=MERCADO, firm_ids=firm_mono).flatten()

prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
sub = prod[prod["market_ids"] == MERCADO].copy()
precio = sub["prices"].to_numpy()
costo_obs = sub["costo"].to_numpy()          # precio mayorista observado

tabla = pd.DataFrame({
    "producto": sub["product_ids"].to_numpy(),
    "marca_madre": sub["firm_ids"].to_numpy(),
    "precio": precio,
    "costo_obs": costo_obs,
    "mc_BLP": costo_blp,
    "mc_logit": costo_logit,
    "markup_BLP_%": 100 * (precio - costo_blp) / precio,
}).set_index("producto")

print("=== Costos marginales, mercado %s (Bertrand multiproducto) ===" % MERCADO)
print(tabla.round(3).to_string())
print("\npromedios:")
print("  precio            : %.3f" % precio.mean())
print("  costo observado   : %.3f" % costo_obs.mean())
print("  mc recuperado BLP : %.3f" % costo_blp.mean())
print("  markup BLP medio  : %.1f%%" % tabla["markup_BLP_%"].mean())
print("\ncorrelacion mc_BLP vs costo observado: %.3f"
      % np.corrcoef(costo_blp, costo_obs)[0, 1])