"""
Script 02: Ejercicio 3, modelo 1.
Logit agregado estimado por MCO con precio y promoción.

Especificación:   ln(s_jt) - ln(s_0t) = beta_0 + beta_1 * precio_jt
                                        + beta_2 * descuento_jt + xi_jt

Correr:  python code/02_logit_mco.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")

y = df["y_logit"]
X = sm.add_constant(df[["precio", "descuento"]])

# errores estandar agrupados por mercado: los xi_jt de un mismo mercado
# comparten shocks (el share del bien externo es comun a todo el mercado)
m1 = sm.OLS(y, X).fit(cov_type="cluster",
                      cov_kwds={"groups": df["mercado"]})
print(m1.summary())

alpha = m1.params["precio"]
print("\nalpha (coef. de precio) = %.4f" % alpha)


