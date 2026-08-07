# src/pinn1d/vi_rayleigh/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def trig_nodal_factor(x, n):
    s1 = torch.sin(math.pi * x)
    sn = torch.sin(n * math.pi * x)
    ratio = sn / (s1 + 1e-12)
    return torch.where(torch.abs(s1) < 1e-6, torch.full_like(x, float(n)), ratio)


class BayesianLinear(nn.Module):
    """
    Capa lineal bayesiana con pesos w ~ N(mu, sigma^2).
    sigma se parametriza como softplus(rho) para garantizar sigma > 0
    sin restringir rho durante la optimización (truco estándar, Blundell et al. 2015).

    Reparametrización: w = mu + sigma * epsilon,  epsilon ~ N(0,1)
    Esto permite backprop a través de una muestra aleatoria.
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Parámetros variacionales de los pesos
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

        # Parámetros variacionales del bias
        self.bias_mu  = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5.0)  # sigma inicial pequeña (~softplus(-5) ≈ 0.0067)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma   = F.softplus(self.bias_rho)

        # Reparametrización
        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias   = self.bias_mu   + bias_sigma   * eps_b

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        KL(q(w) || p(w)) entre q(w) = N(mu, sigma^2) y el prior p(w) = N(0, prior_sigma^2).
        Fórmula cerrada para dos gaussianas (ver Blundell et al. 2015, Ec. 5 / Bishop 2006, Ec. 10.103):

        KL = sum[ log(prior_sigma/sigma) + (sigma^2 + mu^2) / (2*prior_sigma^2) - 0.5 ]
        """
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma   = F.softplus(self.bias_rho)

        def kl_term(mu, sigma):
            prior_sigma = self.prior_sigma
            return torch.sum(
                torch.log(prior_sigma / sigma)
                + (sigma ** 2 + mu ** 2) / (2 * prior_sigma ** 2)
                - 0.5
            )

        return kl_term(self.weight_mu, weight_sigma) + kl_term(self.bias_mu, bias_sigma)


class VIRayleighNet(nn.Module):
    """
    Versión bayesiana de RayleighNet: misma arquitectura y mismo ansatz
    (condiciones de frontera + trig_nodal_factor), pero con pesos
    representados como distribuciones en vez de valores puntuales.

    La energía sigue calculándose vía cociente de Rayleigh (en losses.py),
    ahora sobre una psi muestreada en cada forward pass.
    """
    def __init__(self, n=1, hidden=64, use_sine=True, prior_sigma=1.0):
        super().__init__()
        self.n = n
        self.use_sine = use_sine

        self.fc1 = BayesianLinear(1, hidden, prior_sigma=prior_sigma)
        self.fc2 = BayesianLinear(hidden, hidden, prior_sigma=prior_sigma)
        self.fc3 = BayesianLinear(hidden, 1, prior_sigma=prior_sigma)

    def forward(self, x):
        if self.use_sine:
            z = torch.sin(self.fc1(x))
            z = torch.sin(self.fc2(z))
        else:
            z = torch.tanh(self.fc1(x))
            z = torch.tanh(self.fc2(z))
        out = self.fc3(z)
        Fn = trig_nodal_factor(x, self.n)
        psi = x * (1.0 - x) * Fn * out
        return psi

    def kl_divergence(self):
        """KL total de la red = suma de KL de cada capa bayesiana."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()