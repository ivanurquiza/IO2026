# ============================================================================
#  Réplica de la SLIDE 14 --- "Monte Carlo: infactible vs. factible"
#  Materia: Métodos Econométricos y Organización Industrial Aplicada
# ============================================================================

set.seed(20260524)

## --- Parámetros del experimento --------------------------------------------
mu0 <- 2       # valor verdadero de mu
n   <- 100     # tamaño de muestra
R   <- 4000    # réplicas Monte Carlo

## --- DGP: Gamma con E(y) = mu y Var(y) = 3*mu -------------------------------
# Gamma(shape = k, scale = theta): E = k*theta, Var = k*theta^2.
# Con theta = 3 y k = mu/3  =>  E = mu, Var = 3*mu.
simular <- function(n, mu) rgamma(n, shape = mu / 3, scale = 3)

## --- Momentos (CONSISTENTES: mismo momento en el objetivo y en S) -----------
# g1_i(mu) = y_i - mu
# g2_i(mu) = (y_i - mu)^2 - 3*mu        (dependiente de mu; NO la var. fija)
g_bar <- function(mu, y) c(mean(y - mu), mean((y - mu)^2) - 3 * mu)

Q <- function(mu, y, W) {
  g <- g_bar(mu, y)
  as.numeric(t(g) %*% W %*% g)
}

# Estimador GMM: minimiza la forma cuadrática sobre el escalar mu
est_gmm <- function(y, W) optimize(function(m) Q(m, y, W),
                                   interval = c(0.05, 50))$minimum

# S por observación (misma definición que g_bar), evaluada en un mu dado:
#   Shat = (1/n) sum_i g_i g_i'
S_hat <- function(mu, y) {
  M <- cbind(y - mu, (y - mu)^2 - 3 * mu)
  crossprod(M) / length(y)
}

## --- Matriz de ponderación óptima VERDADERA (caso infactible) ---------------
# S poblacional en mu0 para la Gamma (momentos centrales):
#   S11 = Var(y)        = 3*mu
#   S12 = E[(y-mu)^3]   = 18*mu             (= 2*k*theta^3)
#   S22 = mu4 - sigma^4 = (2 + 18/mu)*(3*mu)^2
S_pob <- function(mu) {
  s11 <- 3 * mu
  s12 <- 18 * mu
  s22 <- (2 + 18 / mu) * (3 * mu)^2
  matrix(c(s11, s12, s12, s22), 2, 2)
}
W_true <- solve(S_pob(mu0))   # W = S^{-1} verdadera (no se estima)

## --- Monte Carlo ------------------------------------------------------------
res <- matrix(NA_real_, R, 4,
              dimnames = list(NULL, c("ybar", "s2_3", "GMM_infact", "GMM_2step")))

for (r in seq_len(R)) {
  y <- simular(n, mu0)

  res[r, "ybar"]       <- mean(y)          # usa solo el momento 1
  res[r, "s2_3"]       <- var(y) / 3       # usa solo el momento 2
  res[r, "GMM_infact"] <- est_gmm(y, W_true)   # GMM con W = S^{-1} verdadera

  # GMM factible TWO-STEP:
  #   Etapa 1: primer paso consistente = ybar (bien portado).
  #   NOTA: arrancar con W = I cae en un mínimo local espurio del objetivo
  #         (es cuártico en mu), por eso se usa ybar como primer paso.
  #   Etapa 2: What = Shat(m1)^{-1}.
  m1 <- mean(y)
  res[r, "GMM_2step"]  <- est_gmm(y, solve(S_hat(m1, y)))
}

## --- Tabla resumen (como en la slide) --------------------------------------
medias  <- unname(colMeans(res))
sd_vec  <- unname(apply(res, 2, sd))
ef_rel  <- (sd_vec[1] / sd_vec)^2          # Var(ybar) / Var(estimador)

tabla <- data.frame(
  Estimador    = c("ybar  (momento 1)",
                   "s^2/3 (momento 2)",
                   "GMM infactible (W = S^-1)",
                   "GMM factible two-step"),
  `E[mu_hat]`  = round(medias, 2),
  `SD`         = round(sd_vec, 2),
  `Efic. rel.` = round(ef_rel, 2),
  check.names  = FALSE
)
cat("\n=== Slide 14: Monte Carlo (n = 100, R = 4000, mu0 = 2) ===\n")
print(tabla, row.names = FALSE)

