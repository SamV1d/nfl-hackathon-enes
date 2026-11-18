"""Estimadores en numpy: base, regresión logística/ridge, árboles, bosque aleatorio y boosting.

Interfaz común: fit(X, y) -> self; predict(X) -> probabilidades (clasificación) o valores
(regresión). X es una matriz float ya codificada (ver features.py).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- modelos base
class MeanBaseline:
    """Predice siempre la media de entrenamiento (tasa base en clasificación)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ------------------------------------------------------------ regresión lineal
class _Standardizer:
    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X):
        return (X - self.mu_) / self.sd_


class RidgeRegression:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.sc_ = _Standardizer().fit(X)
        Z = self.sc_.transform(X)
        self.b0_ = float(np.mean(y))
        yc = y - self.b0_
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.w_ = np.linalg.solve(A, Z.T @ yc)
        return self

    def predict(self, X):
        return self.b0_ + self.sc_.transform(X) @ self.w_


class LogisticRegression:
    """Regresión logística con penalización L2 (Newton-Raphson / IRLS)."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, X, y):
        self.sc_ = _Standardizer().fit(X)
        Z = np.hstack([np.ones((len(X), 1)), self.sc_.transform(X)])
        w = np.zeros(Z.shape[1])
        pen = np.full(Z.shape[1], self.l2)
        pen[0] = 0.0  # el intercepto no se penaliza
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (p - y) + pen * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.w_ = w
        return self

    def predict(self, X):
        Z = np.hstack([np.ones((len(X), 1)), self.sc_.transform(X)])
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w_, -30, 30)))


# --------------------------------------------------------------------- árboles
class _Binner:
    """Cuantiza cada columna en <= n_bins cubetas según cuantiles de entrenamiento."""

    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins

    def fit(self, X):
        self.edges_ = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.n_bins:
                e = (u[:-1] + u[1:]) / 2.0
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges_.append(e)
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.int32)
        for j, e in enumerate(self.edges_):
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
        return out

    @property
    def n_bins_per_feature(self):
        return [len(e) + 1 for e in self.edges_]


class _Tree:
    """Árbol de regresión sobre gradientes/hessianos (hoja = -G/(H+lambda))."""

    def __init__(self, max_depth, min_leaf, lam, mtry, rng):
        self.max_depth, self.min_leaf, self.lam, self.mtry, self.rng = max_depth, min_leaf, lam, mtry, rng

    def fit(self, Xb, g, h, nbins):
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []
        self._build(Xb, g, h, nbins, np.arange(len(g)), 0)
        self.feat = np.array(self.feat)
        self.thr = np.array(self.thr)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.val = np.array(self.val, dtype=float)
        return self

    def _new(self):
        for l in (self.feat, self.thr, self.left, self.right, self.val):
            l.append(0)
        return len(self.feat) - 1

    def _build(self, Xb, g, h, nbins, idx, depth):
        node = self._new()
        G, H = g[idx].sum(), h[idx].sum()
        self.val[node] = -G / (H + self.lam)
        self.feat[node], self.left[node], self.right[node] = -1, -1, -1
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return node
        nfeat = Xb.shape[1]
        feats = np.arange(nfeat)
        if self.mtry is not None and self.mtry < nfeat:
            feats = self.rng.choice(nfeat, self.mtry, replace=False)
        parent = G * G / (H + self.lam)
        best = (1e-9, -1, -1)
        gi, hi = g[idx], h[idx]
        for j in feats:
            nb = nbins[j]
            if nb < 2:
                continue
            col = Xb[idx, j]
            cg = np.cumsum(np.bincount(col, weights=gi, minlength=nb))[:-1]
            ch = np.cumsum(np.bincount(col, weights=hi, minlength=nb))[:-1]
            cn = np.cumsum(np.bincount(col, minlength=nb))[:-1]
            ok = (cn >= self.min_leaf) & (len(idx) - cn >= self.min_leaf)
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = cg**2 / (ch + self.lam) + (G - cg) ** 2 / (H - ch + self.lam) - parent
            gain = np.where(ok, gain, -np.inf)
            b = int(np.argmax(gain))
            if gain[b] > best[0]:
                best = (float(gain[b]), int(j), b)
        if best[1] < 0:
            return node
        _, j, b = best
        go_left = Xb[idx, j] <= b
        self.feat[node], self.thr[node] = j, b
        self.left[node] = self._build(Xb, g, h, nbins, idx[go_left], depth + 1)
        self.right[node] = self._build(Xb, g, h, nbins, idx[~go_left], depth + 1)
        return node

    def predict(self, Xb):
        node = np.zeros(len(Xb), dtype=int)
        for _ in range(self.max_depth + 1):
            f = self.feat[node]
            leaf = f < 0
            if leaf.all():
                break
            fx = np.where(leaf, 0, f)
            go_left = Xb[np.arange(len(Xb)), fx] <= self.thr[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(leaf, node, nxt)
        return self.val[node]


class RandomForest:
    """Bosque aleatorio (bagging + submuestreo de variables). Clasificación y regresión
    usan error cuadrático; en clasificación la media de hojas es la probabilidad."""

    def __init__(self, n_trees=100, max_depth=6, min_leaf=30, mtry_frac=0.4, seed=0):
        self.n_trees, self.max_depth, self.min_leaf = n_trees, max_depth, min_leaf
        self.mtry_frac, self.seed = mtry_frac, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.binner_ = _Binner().fit(X)
        Xb = self.binner_.transform(X)
        nb = self.binner_.n_bins_per_feature
        mtry = max(1, int(round(self.mtry_frac * X.shape[1])))
        self.trees_ = []
        n = len(y)
        for _ in range(self.n_trees):
            s = rng.integers(0, n, n)
            t = _Tree(self.max_depth, self.min_leaf, 0.0, mtry, rng)
            t.fit(Xb[s], -y[s].astype(float), np.ones(n), nb)
            self.trees_.append(t)
        return self

    def predict(self, X):
        Xb = self.binner_.transform(X)
        return np.mean([t.predict(Xb) for t in self.trees_], axis=0)


class GradientBoosting:
    """Boosting con árboles pequeños. loss='logistic' devuelve probabilidades."""

    def __init__(self, loss="logistic", n_trees=150, max_depth=2, learning_rate=0.05,
                 min_leaf=40, lam=5.0, subsample=0.8, seed=0):
        self.loss, self.n_trees, self.max_depth = loss, n_trees, max_depth
        self.lr, self.min_leaf, self.lam, self.subsample, self.seed = learning_rate, min_leaf, lam, subsample, seed

    def fit(self, X, y, eval_set=None):
        rng = np.random.default_rng(self.seed)
        y = y.astype(float)
        self.binner_ = _Binner().fit(X)
        Xb = self.binner_.transform(X)
        nb = self.binner_.n_bins_per_feature
        if self.loss == "logistic":
            p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            self.f0_ = float(np.log(p0 / (1 - p0)))
        else:
            self.f0_ = float(y.mean())
        F = np.full(len(y), self.f0_)
        self.trees_ = []
        n = len(y)
        for _ in range(self.n_trees):
            if self.loss == "logistic":
                p = 1 / (1 + np.exp(-np.clip(F, -30, 30)))
                g, h = p - y, np.maximum(p * (1 - p), 1e-6)
            else:
                g, h = F - y, np.ones(n)
            s = rng.choice(n, int(n * self.subsample), replace=False) if self.subsample < 1 else np.arange(n)
            t = _Tree(self.max_depth, self.min_leaf, self.lam, None, rng)
            t.fit(Xb[s], g[s], h[s], nb)
            F = F + self.lr * t.predict(Xb)
            self.trees_.append(t)
        return self

    def _raw(self, X, n_trees=None):
        Xb = self.binner_.transform(X)
        F = np.full(len(X), self.f0_)
        for t in self.trees_[:n_trees]:
            F = F + self.lr * t.predict(Xb)
        return F

    def predict(self, X, n_trees=None):
        F = self._raw(X, n_trees)
        return 1 / (1 + np.exp(-np.clip(F, -30, 30))) if self.loss == "logistic" else F
