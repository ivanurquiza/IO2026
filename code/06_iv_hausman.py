"""
Script 06: Ejercicio 3, inciso 5.
Modelos 1, 2 y 3 estimados con el instrumento de Hausman.

La idea de Hausman (1996): el precio de un mismo producto en OTROS
mercados sirve como instrumento porque comparte con el precio local los
shocks de costo del fabricante (que son comunes a todos los mercados),
pero no el shock de demanda local xi_jt (que es especifico del mercado).

Construimos el instrumento como el precio promedio de la MISMA marca, en
la MISMA semana, en las OTRAS 72 tiendas ("leave-one-out"):

    z_jt = ( suma de precios de la marca j en la semana t - p_jt ) / 72

Como la consigna admite una lectura alternativa ("precio promedio de
otras marcas en otros mercados"), tambien construimos esa version y
reportamos ambas.

Correr:  python code/06_iv_hausman.py
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

# ---------- instrumento A: misma marca, otras tiendas, misma semana ----
g = df.groupby(["producto", "semana"])["precio"]
df["z_misma"] = (g.transform("sum") - df["precio"]) / (g.transform("size") - 1)

# ---------- instrumento B: otras marcas, otras tiendas, misma semana ---
# "Otras marcas en otros mercados" excluye dos cosas: la marca j en
# cualquier tienda, y todas las marcas del propio mercado. Se construye
# por inclusion-exclusion sobre las sumas de la semana:
#   suma(semana) - suma(marca j en la semana) - suma(propio mercado)
#   + precio propio           <- se resto dos veces, hay que devolverlo
tot = df.groupby("semana")["precio"]
mkt = df.groupby("mercado")["precio"]
num = (tot.transform("sum") - g.transform("sum")
       - mkt.transform("sum") + df["precio"])
den = (tot.transform("size") - g.transform("size")
       - mkt.transform("size") + 1)          # = 803 - 73 - 11 + 1 = 720
df["z_otras"] = num / den

# ---------- absorcion para el modelo 3 ---------------------------------
COLS = ["y_logit", "precio", "descuento", "z_misma", "z_otras"]
dm = df[COLS] - df.groupby("marca_tienda")[COLS].transform("mean")

CL = dict(cov_type="clustered", clusters=df["mercado"])
D = pd.get_dummies(df["producto"], prefix="marca", drop_first=True,
                   dtype=float)


def estimar(z):
    r1 = IV2SLS(df["y_logit"], df[["const", "descuento"]],
                df["precio"], df[z]).fit(**CL)
    r2 = IV2SLS(df["y_logit"], pd.concat([df[["const", "descuento"]], D],
                axis=1), df["precio"], df[z]).fit(**CL)
    r3 = IV2SLS(dm["y_logit"], dm[["descuento"]],
                dm["precio"], dm[z]).fit(**CL)
    return {"Modelo 1": r1, "Modelo 2": r2, "Modelo 3": r3}


for z, etiqueta in [("z_misma", "MISMA marca en otras tiendas"),
                    ("z_otras", "OTRAS marcas en otras tiendas")]:
    print("\n=== Instrumento: %s ===" % etiqueta)
    print("corr(precio, z) global = %.4f | intra marca-tienda = %.4f"
          % (df["precio"].corr(df[z]), dm["precio"].corr(dm[z])))
    for nombre, r in estimar(z).items():
        fs = r.first_stage.diagnostics.loc["precio"]
        print("  %-9s precio = %8.4f (%.4f)   promo = %7.4f (%.4f)"
              "   F = %9.0f   R2 parcial = %.3f"
              % (nombre, r.params["precio"], r.std_errors["precio"],
                 r.params["descuento"], r.std_errors["descuento"],
                 fs["f.stat"], fs["partial.rsquared"]))


# correlacion instrumento-precio en la variacion que usa cada modelo
COLS = ["precio", "costo", "z_misma", "z_otras"]
d1 = df[COLS]
d2 = df[COLS] - df.groupby("producto")[COLS].transform("mean")
d3 = df[COLS] - df.groupby("marca_tienda")[COLS].transform("mean")
print("\n--- corr(precio, instrumento) neta de los EF de cada modelo ---")
for z in ["costo", "z_misma", "z_otras"]:
    print("  %-9s M1 %+.4f   M2 %+.4f   M3 %+.4f"
          % (z, d1["precio"].corr(d1[z]), d2["precio"].corr(d2[z]),
             d3["precio"].corr(d3[z])))