"""
Script 05: Ejercicio 3, inciso 4.
Modelos 1, 2 y 3 estimados por variables instrumentales, usando el costo
del fabricante (proxy del precio mayorista) como instrumento del precio.

En los tres casos:
  endogena   : precio
  instrumento: costo
  exogenas   : descuento (+ efectos fijos segun el modelo)

Los modelos estan exactamente identificados (un instrumento, una
endogena), de modo que no hay test de sobreidentificacion disponible.

Correr:  python code/05_iv_costo.py
"""

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")
df["const"] = 1.0
df["marca_tienda"] = (df["producto"].astype(str) + "-"
                      + df["tienda"].astype(str))

CL = dict(cov_type="clustered", clusters=df["mercado"])


def reportar(nombre, res):        
    fs = res.first_stage.diagnostics.loc["precio"]
    print("\n--- %s ---" % nombre)
    for v in ["precio", "descuento"]:
        print("  %-10s coef = %8.4f   ee = %.4f"
              % (v, res.params[v], res.std_errors[v]))
    print("  primera etapa: F = %.0f   R2 parcial = %.3f"
          % (fs["f.stat"], fs["partial.rsquared"]))


# ---------- Modelo 1: sin efectos fijos --------------------------------
r1 = IV2SLS(df["y_logit"], df[["const", "descuento"]],
            df["precio"], df["costo"]).fit(**CL)
reportar("Modelo 1 - IV", r1)

# ---------- Modelo 2: efectos fijos de marca ---------------------------
D = pd.get_dummies(df["producto"], prefix="marca", drop_first=True,
                   dtype=float)
X2 = pd.concat([df[["const", "descuento"]], D], axis=1)
r2 = IV2SLS(df["y_logit"], X2, df["precio"], df["costo"]).fit(**CL)
reportar("Modelo 2 - IV (EF de marca)", r2)

# ---------- Modelo 3: efectos fijos de marca-tienda, absorbidos --------
# IMPORTANTE: por Frisch-Waugh-Lovell para IV hay que residualizar TODAS
# las variables con la misma proyeccion, incluido el INSTRUMENTO. Omitir
# el instrumento produce un estimador distinto e inconsistente.
COLS = ["y_logit", "precio", "descuento", "costo"]
dm = df[COLS] - df.groupby("marca_tienda")[COLS].transform("mean")
r3 = IV2SLS(dm["y_logit"], dm[["descuento"]],
            dm["precio"], dm["costo"]).fit(**CL)
reportar("Modelo 3 - IV (EF de marca-tienda)", r3)

# ---------- diagnostico del instrumento --------------------------------
print("\n--- relevancia del instrumento ---")
print("corr(precio, costo) global      = %.4f"
      % df[["precio", "costo"]].corr().iloc[0, 1])
print("corr(precio, costo) intra grupo = %.4f"
      % dm[["precio", "costo"]].corr().iloc[0, 1])
