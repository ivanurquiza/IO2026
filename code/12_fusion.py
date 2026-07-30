"""
Script 12: Ejercicio 5a. Analisis de fusion (modelo logit).

Se fusionan las tres primeras marcas madre. Bajo Bertrand-Nash, la
fusion no cambia los costos marginales pero si la matriz de propiedad H:
las marcas 1, 2 y 3 pasan a tener un unico dueno. Al internalizar la
competencia entre sus productos, la firma fusionada tiene incentivo a
subir precios (efecto unilateral).

Procedimiento (tienda 9, semana 10):
  1. Recuperar los costos marginales con la propiedad PRE-fusion.
  2. Verificar que, con esa misma propiedad, compute_prices reproduce
     los precios observados (control de consistencia que pide la consigna).
  3. Recomputar los precios con la propiedad POST-fusion.

Correr:  python code/12_fusion.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

pyblp.options.verbose = False
RAIZ = Path(__file__).resolve().parent.parent
MERCADO = "9-10"

# ---------- demanda logit (dispersion nula) ----------------------------
with open(RAIZ / "output" / "blp_medios.pkl", "rb") as f:
    res_blp = pickle.load(f)
res_logit = res_blp.problem.solve(
    sigma=np.zeros((5, 5)), pi=np.zeros((5, 1)),
    method="1s", optimization=pyblp.Optimization("return"))

# ---------- 1. costos con propiedad PRE-fusion -------------------------
# firm_ids ya son las 4 marcas madre en la base -> esta es la propiedad
# de partida (marcas 1, 2, 3 y 4 como duenos separados).
costo = res_logit.compute_costs(market_id=MERCADO).flatten()

prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
sub = prod[prod["market_ids"] == MERCADO].copy()
precio_obs = sub["prices"].to_numpy()

# ---------- 2. VERIFICACION: precios con propiedad PRE-fusion ----------
# Recalcular precios con la MISMA propiedad con la que se recuperaron los
# costos debe devolver los precios observados. Es el control que pide la
# consigna.
precio_pre = res_logit.compute_prices(
    costs=costo, market_id=MERCADO).flatten()
print("=== Verificacion (sin fusion) ===")
print("maxima diferencia con el precio observado: %.2e"
      % np.abs(precio_pre - precio_obs).max())

# ---------- 3. propiedad POST-fusion ----------------------------------
# Las marcas 1, 2 y 3 pasan a ser un unico dueno; la 4 sigue sola.
firm_post = pd.Series(sub["firm_ids"].to_numpy()).replace(
    {2: 1, 3: 1}).to_numpy()
precio_post = res_logit.compute_prices(
    costs=costo, firm_ids=firm_post, market_id=MERCADO).flatten()

# ---------- tabla ------------------------------------------------------
tabla = pd.DataFrame({
    "producto": sub["product_ids"].to_numpy(),
    "marca": sub["firm_ids"].to_numpy(),
    "costo_mg": costo,
    "precio_obs": precio_obs,
    "precio_post": precio_post,
    "cambio_%": 100 * (precio_post - precio_obs) / precio_obs,
}).set_index("producto")

print("\n=== Precios post-fusion (marcas 1-2-3), mercado %s ===" % MERCADO)
print(tabla.round(3).to_string())
print("\ncambio de precio promedio:")
print("  marcas fusionadas (1-3): %+.1f%%"
      % tabla.loc[tabla["marca"].isin([1, 2, 3]), "cambio_%"].mean())
print("  marca 4 (independiente): %+.1f%%"
      % tabla.loc[tabla["marca"] == 4, "cambio_%"].mean())