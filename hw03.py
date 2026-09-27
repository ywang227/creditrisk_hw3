import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given parameters
mu = 0.16905
sigma = 0.4907
r = 0.0011888
t = 0.0
T = 0.291667
Delta = 10 / 252
S0 = 152.51
K = 170.0
M = 100
N = 100_000

def put_price(time, stock_price):
    """Black-Scholes price of a European put."""
    tau = T - time
    d1 = (np.log(stock_price / K)
          + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return (K * np.exp(-r * tau) * norm.cdf(-d2)
            - stock_price * norm.cdf(-d1))

tau0 = T - t
d1_0 = (np.log(S0 / K)
        + (r + 0.5 * sigma**2) * tau0) / (sigma * np.sqrt(tau0))
d2_0 = d1_0 - sigma * np.sqrt(tau0)

delta0 = norm.cdf(d1_0) - 1
gamma0 = norm.pdf(d1_0) / (S0 * sigma * np.sqrt(tau0))
theta0 = (-sigma * S0 * norm.pdf(d1_0) / (2 * np.sqrt(tau0))
          + K * r * np.exp(-r * tau0) * norm.cdf(-d2_0))

# Under the physical measure:
# X = log(S_{t+Delta}/S_t)
#   ~ Normal((mu - sigma^2/2) Delta, sigma^2 Delta)
rng = np.random.default_rng(731)
x = ((mu - 0.5 * sigma**2) * Delta
     + sigma * np.sqrt(Delta) * rng.standard_normal(N))
S_next = S0 * np.exp(x)

loss_full = M * (
    put_price(t + Delta, S_next)
    - put_price(t, S0)
    - delta0 * S0 * np.expm1(x)
)

loss_linear = np.full(N, M * theta0 * Delta)

loss_quadratic = (
    M * theta0 * Delta
    + 0.5 * M * S0**2 * gamma0 * x**2
)

for name, losses in [
    ("Full", loss_full),
    ("Linearized", loss_linear),
    ("Second order", loss_quadratic),
]:
    print(
        f"{name:12s} "
        f"mean={losses.mean():9.2f}  "
        f"std={losses.std():9.2f}  "
        f"5%={np.quantile(losses, 0.05):9.2f}  "
        f"median={np.median(losses):9.2f}  "
        f"95%={np.quantile(losses, 0.95):9.2f}"
    )

# Put the continuous distributions on a common scale.
# The upper cutoff only affects display, not the calculations above.
right_edge = max(
    np.quantile(loss_full, 0.995),
    np.quantile(loss_quadratic, 0.995),
)
bins = np.linspace(min(loss_full.min(), loss_quadratic.min()) - 5,
                   right_edge, 110)

plt.figure(figsize=(10, 5))
plt.hist(loss_full, bins=bins, density=True, alpha=0.55,
         label="Full loss")
plt.hist(loss_quadratic, bins=bins, density=True, alpha=0.45,
         label="Second-order loss")
plt.axvline(loss_linear[0], color="black", linestyle="--",
            linewidth=2, label="Linearized loss (point mass)")
plt.xlabel("Portfolio loss ($)")
plt.ylabel("Estimated probability density")
plt.title("Ten-day portfolio loss: 100,000 simulations")
plt.legend()
plt.tight_layout()
plt.show()