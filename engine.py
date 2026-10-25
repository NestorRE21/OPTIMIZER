"""
Motor de cálculo Black-Litterman + Optimización de frontera eficiente.
Archivo autocontenido: config + funciones de cálculo en un solo módulo.
"""

# ── Temporales ────────────────────────────────────────────────────────────────
TRADING_DAYS             = 252
INVESTMENT_HORIZON_YEARS = 1
INVESTMENT_HORIZON_DAYS  = TRADING_DAYS * INVESTMENT_HORIZON_YEARS
RF_ANNUAL                = 0.02
RF_DAILY                 = (1.0 + RF_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

# ── Límites de concentración ──────────────────────────────────────────────────
MIN_WEIGHT        = 0.00
MAX_WEIGHT_EQUITY = 0.20
MAX_WEIGHT_FICO   = 1.00
MIN_POSITION      = 0.02

# ── Cardinalidad ──────────────────────────────────────────────────────────────
MIN_EQUITY_ASSETS = 4
MAX_EQUITY_ASSETS = 4
MIN_FICO_ASSETS   = 1
MAX_FICO_ASSETS   = 1

# ── Calidad de datos ──────────────────────────────────────────────────────────
MAX_STALE_DAYS       = 5
MAX_MISSING_PCT      = 0.10
MAX_MISSING_PCT_LATE = 0.99
LATE_START_ASSETS    = ["AUNA PE Equity"]
WINSORIZE_LIMIT      = 0.005

# ── Optimización ──────────────────────────────────────────────────────────────
TARGET_TOLERANCE            = 0.04
MAX_CONSERVATIVE_VOL_ANNUAL = 0.06
MIN_SECTOR_EXPOSURE         = 0.05
MAX_SECTOR_WEIGHT_EQUITY    = 0.25
EPS                         = 1e-10

# ── Perfiles (3 base) ────────────────────────────────────────────────────────
PROFILE_TARGETS = {
    "Conservador": {"equity": 0.30, "fico": 0.70},
    "Moderado":    {"equity": 0.50, "fico": 0.50},
    "Agresivo":    {"equity": 0.70, "fico": 0.30},
}

PROFILE_MAX_DRAWDOWN = {
    "Conservador": 0.20,
    "Moderado":    0.30,
    "Agresivo":    0.45,
}

PROFILE_MIN_BETA = {
    "Conservador": 0.40,
    "Moderado":    0.60,
    "Agresivo":    0.80,
}

PROFILE_MAX_BETA = {
    "Conservador": 0.90,
    "Moderado":    1.10,
    "Agresivo":    1.90,
}

PROFILE_EXCLUSIONS = {
    "Conservador": [
        "TSLA PE Equity", "META PE Equity", "HBM PE Equity",
        "NVDA PE Equity", "BVN PE Equity",  "AMZN PE Equity",
    ],
    "Moderado": ["TSLA PE Equity", "META PE Equity", "HBM PE Equity"],
    "Agresivo": ["TSLA PE Equity", "META PE Equity"],
}

# ── 5 perfiles (activos fijos) ───────────────────────────────────────────────
FIXED_PROFILES = {
    "Perfil A (30/70)": {"equity": 0.30, "fico": 0.70},
    "Perfil B (40/60)": {"equity": 0.40, "fico": 0.60},
    "Perfil C (50/50)": {"equity": 0.50, "fico": 0.50},
    "Perfil D (60/40)": {"equity": 0.60, "fico": 0.40},
    "Perfil E (70/30)": {"equity": 0.70, "fico": 0.30},
}

FIVE_PROFILE_BETA_MAP = {
    "Perfil A (30/70)": "Conservador",
    "Perfil B (40/60)": "Conservador",
    "Perfil C (50/50)": "Moderado",
    "Perfil D (60/40)": "Moderado",
    "Perfil E (70/30)": "Agresivo",
}

# ── FICO forzado ──────────────────────────────────────────────────────────────
FICO_USD = {
    "FICCMP13": {
        "ret_annual": 0.06, "vol_annual": 0.0100, "beta": 0.30,
        "sector": "Factoring", "region": "Perú", "moneda": "USD",
        "instrumento": "Fondo de inversión", "asset_class": "Renta Fija",
    },
}

FICO_PEN = {
    "FICRPS11": {
        "ret_annual": 0.07, "vol_annual": 0.0080, "beta": 0.30,
        "sector": "Factoring", "region": "Perú", "moneda": "PEN",
        "instrumento": "Fondo de inversión", "asset_class": "Renta Fija",
    },
}

FICO_FORCED = FICO_USD   # ← cambiar a FICO_PEN para soles

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_TICKER = "^GSPC"



import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf



# ═════════════════════════════════════════════════════════════════════════════
# HELPERS NUMÉRICOS
# ═════════════════════════════════════════════════════════════════════════════

def nearest_psd(matrix, epsilon=1e-8):
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (psd + psd.T)


def solve_psd_system(A, B, ridge=1e-8):
    A = nearest_psd(A, epsilon=ridge)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


def calculate_max_drawdown(log_returns):
    ret = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return np.nan
    wealth = np.exp(ret.cumsum())
    drawdown = wealth / wealth.cummax() - 1.0
    return abs(float(drawdown.min()))


def calculate_drawdown(log_ret_series):
    ret = log_ret_series.replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        raise ValueError("Serie vacía.")
    wealth = np.exp(ret.cumsum())
    return wealth / wealth.cummax() - 1.0


def cumulative_return_for_period(log_ret_series, days):
    ret = log_ret_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) == 0:
        return np.nan
    ret = ret.iloc[-min(days, len(ret)):]
    return np.exp(ret.sum()) - 1.0


def portfolio_historical_returns(weights, returns):
    missing = sorted(set(weights.index) - set(returns.columns))
    if missing:
        raise ValueError(f"Activos sin serie: {missing}")
    if weights.sum() < EPS:
        raise ValueError("Pesos suman cero.")
    w = weights.reindex(returns.columns).fillna(0.0)
    w = w / w.sum()
    return returns @ w


def sector_exposure_from_weights(weights, asset_metadata, sector_col="sector"):
    if weights.sum() < EPS:
        raise ValueError("Pesos suman cero.")
    w = weights / weights.sum()
    meta = asset_metadata.reindex(w.index).copy()
    meta[sector_col] = meta[sector_col].fillna("Sin clasificar")
    meta["peso"] = w
    return meta.groupby(sector_col)["peso"].sum().sort_values(ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS — TODO DESDE YAHOO FINANCE
# ═════════════════════════════════════════════════════════════════════════════

def _remove_timezone(index):
    index = pd.to_datetime(index)
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def download_equity_prices(tickers, period="5y"):
    """Descarga precios de cierre ajustados de yfinance para una lista de tickers."""
    if not tickers:
        raise ValueError("Lista de tickers vacía.")

    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"Sin datos de yfinance para: {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers if len(tickers) == 1 else prices.columns

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.index = _remove_timezone(prices.index)
    prices = prices.sort_index()
    prices = prices.replace(0, np.nan)
    prices[prices < 0] = np.nan
    prices = prices.ffill(limit=MAX_STALE_DAYS).dropna(how="all")

    # Descartar tickers con demasiados NaN
    miss_pct = prices.isna().mean()
    valid = [t for t in miss_pct.index if miss_pct[t] <= MAX_MISSING_PCT]
    dropped = [t for t in miss_pct.index if t not in valid]
    if dropped:
        print(f"  [WARN] Tickers descartados por exceso NaN: {dropped}")
    prices = prices[valid].dropna(how="all")

    if prices.empty:
        raise ValueError("Todos los tickers descartados por datos insuficientes.")

    return prices


def build_equity_metadata(tickers):
    """Construye metadata (sector, beta, moneda, región) desde yfinance .info."""
    meta_rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            meta_rows.append({
                "asset": ticker,
                "sector": info.get("sector", "Sin clasificar"),
                "region": info.get("country", "Sin clasificar"),
                "moneda": info.get("currency", "USD"),
                "instrumento": info.get("quoteType", "EQUITY"),
                "beta": info.get("beta", np.nan),
                "asset_class": "Renta Variable",
                "fuente": "yfinance",
            })
        except Exception:
            meta_rows.append({
                "asset": ticker, "sector": "Sin clasificar",
                "region": "Sin clasificar", "moneda": "USD",
                "instrumento": "EQUITY", "beta": np.nan,
                "asset_class": "Renta Variable", "fuente": "yfinance",
            })
    return pd.DataFrame(meta_rows).set_index("asset")


def calculate_returns(price_data):
    """Calcula log-returns diarios con winsorización."""
    prices = price_data.copy().apply(pd.to_numeric, errors="coerce")
    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if WINSORIZE_LIMIT and WINSORIZE_LIMIT > 0:
        lower = log_ret.quantile(WINSORIZE_LIMIT, axis=0)
        upper = log_ret.quantile(1.0 - WINSORIZE_LIMIT, axis=0)
        log_ret = log_ret.clip(lower=lower, upper=upper, axis=1)

    col_nan = log_ret.isna().mean()
    bad = [c for c in col_nan.index if col_nan[c] > MAX_MISSING_PCT]
    if bad:
        log_ret = log_ret.drop(columns=bad)
    return log_ret.dropna(how="all").fillna(0.0)


def build_synthetic_fico(fico_forced, date_index, seed=42):
    """Genera log-returns sintéticos para FICO con la vol declarada."""
    rng = np.random.default_rng(seed)
    frames, meta_rows = {}, []
    for ticker, params in fico_forced.items():
        daily_log_ret = np.log(1.0 + params["ret_annual"]) / TRADING_DAYS
        daily_vol = params["vol_annual"] / np.sqrt(TRADING_DAYS)
        noise = rng.normal(0, daily_vol, size=len(date_index))
        frames[ticker] = pd.Series(daily_log_ret + noise, index=date_index, name=ticker)
        meta_rows.append({
            "asset": ticker, "region": params["region"], "sector": params["sector"],
            "moneda": params["moneda"], "instrumento": params["instrumento"],
            "beta": params["beta"], "asset_class": params["asset_class"], "fuente": "forzado",
        })
    return pd.DataFrame(frames).loc[date_index], pd.DataFrame(meta_rows).set_index("asset")


def download_benchmark(ticker=BENCHMARK_TICKER, start=None, end=None):
    """Descarga log-returns diarios del benchmark."""
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"Sin datos para '{ticker}'.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prices = raw["Close"].copy()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices.index = _remove_timezone(prices.index)
    prices = prices.sort_index().dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()
    log_ret.name = ticker
    return log_ret


def load_full_universe(equity_tickers, fico_forced=None, period="5y"):
    """Pipeline completo: yfinance → returns → FICO → benchmark → alineación."""
    if fico_forced is None:
        fico_forced = FICO_FORCED

    # 1. Descargar precios RV desde yfinance
    eq_prices = download_equity_prices(equity_tickers, period=period)

    # 2. Metadata desde yfinance .info
    eq_meta = build_equity_metadata(list(eq_prices.columns))

    # 3. Log-returns RV
    rets_eq = calculate_returns(eq_prices)

    # 4. FICO sintético
    rets_fi, fi_meta = build_synthetic_fico(fico_forced, rets_eq.index)

    # 5. Consolidar
    returns = pd.concat([rets_eq, rets_fi], axis=1, join="outer")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)

    equity_assets = [a for a in eq_prices.columns if a in returns.columns]
    fico_assets = list(fico_forced.keys())
    asset_metadata = pd.concat([eq_meta, fi_meta], axis=0).reindex(returns.columns)

    # 6. Benchmark
    idx = _remove_timezone(pd.to_datetime(returns.index))
    start = (idx.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (idx.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    bmk_ret = download_benchmark(start=start, end=end)

    # 7. Alinear
    returns.index = _remove_timezone(returns.index)
    bmk_ret.index = _remove_timezone(bmk_ret.index)
    returns, bmk_ret = returns.align(bmk_ret, join="inner", axis=0)

    return returns, equity_assets, fico_assets, asset_metadata, bmk_ret


def get_market_cap_weights(assets, equity_assets, fico_assets, equity_target, fico_target):
    weights = {}
    for a in equity_assets:
        weights[a] = equity_target / len(equity_assets)
    for a in fico_assets:
        weights[a] = fico_target / len(fico_assets)
    w = pd.Series(weights).reindex(assets).fillna(0.0)
    return w / w.sum()


# ═════════════════════════════════════════════════════════════════════════════
# COVARIANZA + CALIBRACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def estimate_covariance(rets, ridge=1e-8):
    rets_clean = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    lw = LedoitWolf()
    lw.fit(rets_clean.to_numpy(dtype=float))
    cov_annual = nearest_psd(lw.covariance_ * TRADING_DAYS, epsilon=ridge)
    return cov_annual, lw.shrinkage_


def calibrate_market(rets, benchmark_returns, tau_method=0.05,
                     min_lambda=0.50, max_lambda=10.00, fallback_lambda=2.50):
    """Calibra tau y lambda para BL."""
    rets = rets.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    bmk = benchmark_returns.copy().replace([np.inf, -np.inf], np.nan).dropna()
    common = rets.index.intersection(bmk.index)
    rets_al, bmk_al = rets.loc[common], bmk.loc[common]
    n_obs = len(rets_al)

    if tau_method == "sample_size":
        tau = 1.0 / np.sqrt(n_obs)
    else:
        tau = float(tau_method)

    ret_ann = bmk_al.mean() * TRADING_DAYS
    var_ann = bmk_al.var(ddof=1) * TRADING_DAYS
    vol_ann = np.sqrt(var_ann)

    raw_lambda = (ret_ann - RF_ANNUAL) / var_ann
    if np.isfinite(raw_lambda) and min_lambda <= raw_lambda <= max_lambda:
        lambda_mkt = raw_lambda
    else:
        lambda_mkt = fallback_lambda

    return {
        "tau": tau, "lambda": lambda_mkt, "lambda_raw": raw_lambda,
        "bmk_ret_ann": ret_ann, "bmk_vol_ann": vol_ann,
        "sharpe_bmk": (ret_ann - RF_ANNUAL) / vol_ann if vol_ann > EPS else np.nan,
        "n_obs": n_obs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BLACK-LITTERMAN
# ═════════════════════════════════════════════════════════════════════════════

def build_views(assets, views_config):
    """Construye P, Q, c, k desde la config de views."""
    assets = list(assets)
    idx_map = {a: i for i, a in enumerate(assets)}
    n = len(assets)
    P_rows, Q_vals, c_vals, k_vals, names = [], [], [], [], []

    for i, v in enumerate(views_config):
        p = np.zeros(n)
        vtype = v["type"]
        vname = v.get("name", f"View_{i+1}")

        if vtype == "absolute":
            p[idx_map[v["asset"]]] = 1.0
        elif vtype == "relative":
            p[idx_map[v["long"]]] = 1.0
            p[idx_map[v["short"]]] = -1.0

        P_rows.append(p)
        Q_vals.append(float(v["q"]))
        c_vals.append(float(v.get("c", 1.0)))
        k_vals.append(float(v.get("k", 1.0)))
        names.append(vname)

    return (np.vstack(P_rows), np.array(Q_vals), np.array(c_vals), np.array(k_vals), names)


def black_litterman_posterior(rets, P, Q, tau, risk_aversion,
                               market_weights=None, c=1.0, k=1.0, ridge=1e-8):
    """Distribución posterior BL completa."""
    rets = rets.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    asset_names = list(rets.columns)
    n = len(asset_names)
    P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float).reshape(-1)
    n_views = P.shape[0]

    cov_matrix, shrinkage = estimate_covariance(rets, ridge=ridge)

    if market_weights is None:
        market_weights = np.ones(n) / n
    else:
        market_weights = np.asarray(market_weights, dtype=float).reshape(-1)
        market_weights = market_weights / market_weights.sum()

    pi_excess = risk_aversion * (cov_matrix @ market_weights)

    view_net_exp = P @ np.ones(n)
    Q_excess = Q - view_net_exp * RF_ANNUAL

    view_cov = nearest_psd(P @ cov_matrix @ P.T, epsilon=ridge)
    c_vec = np.full(n_views, float(c)) if np.isscalar(c) else np.asarray(c, dtype=float)
    k_vec = np.full(n_views, float(k)) if np.isscalar(k) else np.asarray(k, dtype=float)
    omega_matrix = nearest_psd(np.diag((k_vec / c_vec) * tau * np.diag(view_cov)), epsilon=ridge)

    tau_cov = nearest_psd(tau * cov_matrix, epsilon=ridge)
    inv_tau_cov = solve_psd_system(tau_cov, np.eye(n), ridge=ridge)
    inv_omega = solve_psd_system(omega_matrix, np.eye(n_views), ridge=ridge)

    post_prec = nearest_psd(inv_tau_cov + P.T @ inv_omega @ P, epsilon=ridge)
    M = solve_psd_system(post_prec, np.eye(n), ridge=ridge)
    mu_bl_excess = M @ (inv_tau_cov @ pi_excess + P.T @ inv_omega @ Q_excess)
    sigma_bl = nearest_psd(cov_matrix + M, epsilon=ridge)

    ret_bl = mu_bl_excess + RF_ANNUAL
    sd_bl = np.sqrt(np.diag(sigma_bl))
    sharpe_bl = (ret_bl - RF_ANNUAL) / (sd_bl + EPS)

    return {
        "cov_matrix": pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names),
        "sigma_bl": pd.DataFrame(sigma_bl, index=asset_names, columns=asset_names),
        "pi_excess": pd.Series(pi_excess, index=asset_names),
        "mu_bl_excess": pd.Series(mu_bl_excess, index=asset_names),
        "ret_bl": pd.Series(ret_bl, index=asset_names),
        "sd_bl": pd.Series(sd_bl, index=asset_names),
        "sharpe_bl": pd.Series(sharpe_bl, index=asset_names),
        "delta_views": pd.Series(mu_bl_excess - pi_excess, index=asset_names),
        "market_weights": pd.Series(market_weights, index=asset_names),
        "shrinkage": shrinkage,
        "tau": tau, "risk_aversion": risk_aversion,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCREENING + FRONTERA
# ═════════════════════════════════════════════════════════════════════════════

def select_equity_for_profile(profile_name, mu, cov, asset_names, asset_metadata,
                               min_assets=MIN_EQUITY_ASSETS, max_assets=MAX_EQUITY_ASSETS):
    asset_names = list(asset_names)
    excluidos = PROFILE_EXCLUSIONS.get(profile_name, [])
    asset_names = [a for a in asset_names if a not in excluidos]
    n = len(asset_names)
    if n <= max(min_assets, 1):
        return asset_names

    mu_s = mu.loc[asset_names] if isinstance(mu, pd.Series) else pd.Series(mu, index=asset_names)
    cov_np = nearest_psd(cov.loc[asset_names, asset_names].to_numpy(dtype=float) if isinstance(cov, pd.DataFrame)
                         else cov)
    vol = np.sqrt(np.maximum(np.diag(cov_np), EPS))
    corr = np.clip(cov_np / np.outer(vol, vol), -1.0, 1.0)
    avg_corr = pd.Series(np.nanmean(np.where(np.eye(n), np.nan, corr), axis=1), index=asset_names).fillna(0.0)
    sharpe = pd.Series((mu_s.to_numpy(dtype=float) - RF_ANNUAL) / vol, index=asset_names)
    beta_ind = pd.to_numeric(asset_metadata.reindex(asset_names)["beta"], errors="coerce").fillna(1.0)

    if profile_name == "Conservador":
        ws, wr, wd, wb = 0.10, 0.15, 0.50, 0.25
    elif profile_name == "Moderado":
        ws, wr, wd, wb = 0.15, 0.25, 0.40, 0.20
    else:
        ws, wr, wd, wb = 0.10, 0.55, 0.30, 0.05

    def prank(s):
        return s.replace([np.inf, -np.inf], np.nan).rank(pct=True)

    score = (ws * prank(sharpe) + wr * prank(mu_s) + wd * prank(-avg_corr) + wb * prank(-beta_ind)).fillna(0.0)
    return score.sort_values(ascending=False).head(max(min_assets, min(n, max_assets))).index.tolist()


def optimize_frontier(mu, cov, asset_names, equity_assets, fico_assets,
                      equity_target, fico_target, profile_name, asset_metadata,
                      num_puntos=120, ridge=1e-8):
    """Construye frontera eficiente para un perfil con todas las restricciones."""
    asset_names = list(asset_names)
    equity_set, fico_set = set(equity_assets), set(fico_assets)
    n = len(asset_names)
    n_eq, n_fi = len(equity_assets), len(fico_assets)

    mu_s = mu.loc[asset_names] if isinstance(mu, pd.Series) else pd.Series(mu, index=asset_names)
    cov_df = cov.loc[asset_names, asset_names] if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov, index=asset_names, columns=asset_names)
    mu_np = mu_s.to_numpy(dtype=float)
    cov_np = nearest_psd(cov_df.to_numpy(dtype=float), epsilon=ridge)

    bounds = []
    for a in asset_names:
        if a in equity_set:
            bounds.append((0.0, MAX_WEIGHT_EQUITY))
        elif a in fico_set:
            bounds.append((fico_target, fico_target) if n_fi == 1 else (0.0, MAX_WEIGHT_FICO))
        else:
            raise ValueError(f"Activo sin clase: '{a}'")

    eq_idx = np.array([i for i, a in enumerate(asset_names) if a in equity_set])
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if n_fi > 1:
        constraints.append({"type": "eq", "fun": lambda w, idx=eq_idx: w[idx].sum() - equity_target})

    # Beta
    beta_vector = pd.to_numeric(asset_metadata.reindex(asset_names)["beta"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    constraints.append({"type": "ineq", "fun": lambda w, bv=beta_vector, bmax=PROFILE_MAX_BETA[profile_name]: bmax - float(w @ bv)})
    constraints.append({"type": "ineq", "fun": lambda w, bv=beta_vector, bmin=PROFILE_MIN_BETA[profile_name]: float(w @ bv) - bmin})

    # Sector
    sector_series = asset_metadata.reindex(asset_names)["sector"].fillna("Sin clasificar")
    sectores_rv = sector_series[pd.Series(asset_names).isin(equity_set).values].unique()
    for sector in sectores_rv:
        s_idx = np.array([i for i, a in enumerate(asset_names) if a in equity_set and sector_series.loc[a] == sector])
        if len(s_idx) > 0:
            constraints.append({"type": "ineq", "fun": lambda w, idx=s_idx: MAX_SECTOR_WEIGHT_EQUITY - w[idx].sum()})

    # x0
    x0 = np.zeros(n)
    fi_idx_list = [i for i, a in enumerate(asset_names) if a in fico_set]
    eq_idx_list = [i for i, a in enumerate(asset_names) if a in equity_set]
    if n_fi == 1:
        x0[fi_idx_list[0]] = fico_target
    else:
        for i in fi_idx_list: x0[i] = fico_target / n_fi
    for i in eq_idx_list: x0[i] = equity_target / n_eq

    opts = {"maxiter": 5000, "ftol": 1e-10, "disp": False}
    r_min_res = minimize(lambda w: float(w @ mu_np), x0, method="SLSQP", bounds=bounds, constraints=constraints, options=opts)
    r_max_res = minimize(lambda w: -float(w @ mu_np), x0, method="SLSQP", bounds=bounds, constraints=constraints, options=opts)

    if not r_min_res.success or not r_max_res.success:
        raise RuntimeError(f"Rango factible no encontrado para {profile_name}.")

    r_min, r_max = float(r_min_res.x @ mu_np), float(r_max_res.x @ mu_np)
    if abs(r_max - r_min) < EPS:
        raise RuntimeError("Rango de retornos ≈ 0.")

    levels = np.linspace(r_min, r_max * 0.995 + r_min * 0.005, num_puntos)
    frontier_pts, weights_list = [], []
    x_start = x0.copy()

    for r in levels:
        cons_r = constraints + [{"type": "eq", "fun": lambda w, _r=r: float(w @ mu_np) - _r}]
        res = minimize(lambda w: float(w.T @ cov_np @ w), x_start, method="SLSQP",
                       bounds=bounds, constraints=cons_r, options=opts)
        if res.success:
            w = np.asarray(res.x, dtype=float)
            w[np.abs(w) < 1e-10] = 0.0
            w = w / w.sum()
            frontier_pts.append([float(w @ mu_np), float(np.sqrt(max(w.T @ cov_np @ w, EPS)))])
            weights_list.append(w)
            x_start = w

    if len(frontier_pts) < 5:
        raise RuntimeError(f"Solo {len(frontier_pts)} puntos válidos.")

    frontier = np.asarray(frontier_pts)
    weights = np.asarray(weights_list)
    order = np.argsort(frontier[:, 0])
    return frontier[order], weights[order]


def pick_portfolio(profile_name, frontier, weights_frontier, asset_names,
                   equity_assets, fico_assets, target_equity, target_fico,
                   returns, asset_metadata):
    """Selecciona el mejor portafolio de la frontera para un perfil."""
    equity_set = set(equity_assets)
    records = []

    for i, w in enumerate(weights_frontier):
        ret, risk = float(frontier[i, 0]), float(frontier[i, 1])
        sharpe = (ret - RF_ANNUAL) / max(risk, EPS)
        w_s = pd.Series(w, index=asset_names)
        eq_w = float(w_s[w_s.index.isin(equity_assets)].sum())
        hist = portfolio_historical_returns(weights=w_s, returns=returns)
        max_dd = calculate_max_drawdown(hist)
        records.append({"idx": i, "return": ret, "risk": risk, "sharpe": sharpe,
                        "equity_weight": eq_w, "max_drawdown": max_dd})

    cands = pd.DataFrame(records)
    max_dd_allowed = PROFILE_MAX_DRAWDOWN[profile_name]
    cands["drawdown_ok"] = cands["max_drawdown"] <= max_dd_allowed + 1e-12
    cands["vol_ok"] = cands["risk"] <= MAX_CONSERVATIVE_VOL_ANNUAL + 1e-12 if profile_name == "Conservador" else True
    cands["all_ok"] = cands["drawdown_ok"] & cands["vol_ok"]

    feasible = cands[cands["all_ok"]].copy()
    if feasible.empty:
        feasible = cands.sort_values("max_drawdown").head(3).copy()

    if profile_name == "Conservador":
        selected = feasible.sort_values("risk").iloc[0]
    elif profile_name == "Agresivo":
        selected = feasible.sort_values("return", ascending=False).iloc[0]
    else:
        selected = feasible.sort_values("sharpe", ascending=False).iloc[0]

    idx = int(selected["idx"])
    w = pd.Series(weights_frontier[idx], index=asset_names, name=profile_name)

    # Post-limpieza: forzar buckets
    w_eq = w[w.index.isin(equity_set)].copy()
    w_fi = w[~w.index.isin(equity_set)].copy()
    w_eq = w_eq[w_eq >= 0.001]
    if len(w_eq) < MIN_EQUITY_ASSETS:
        w_eq = w[w.index.isin(equity_set)].nlargest(MIN_EQUITY_ASSETS)
    w_fi = w_fi[w_fi > EPS]
    if w_eq.sum() > EPS:
        w_eq = w_eq / w_eq.sum() * target_equity
    if w_fi.sum() > EPS:
        w_fi = w_fi / w_fi.sum() * target_fico
    w = pd.concat([w_eq, w_fi])
    w.name = profile_name
    if w.sum() > EPS:
        w = w / w.sum()

    beta_vec = pd.to_numeric(asset_metadata.reindex(w.index)["beta"], errors="coerce").fillna(1.0)

    return {
        "profile": profile_name,
        "idx": idx,
        "weights": w.sort_values(ascending=False),
        "return": float(selected["return"]),
        "risk": float(selected["risk"]),
        "sharpe": float(selected["sharpe"]),
        "equity_weight": float(w[w.index.isin(equity_assets)].sum()),
        "fico_weight": float(w[~w.index.isin(equity_assets)].sum()),
        "max_drawdown": float(selected["max_drawdown"]),
        "beta": float((w * beta_vec).sum()),
        "frontier": frontier,
        "weights_frontier": weights_frontier,
        "frontier_assets": asset_names,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def performance_metrics(log_returns, benchmark=None):
    ret = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    total_ret = np.exp(ret.sum()) - 1.0
    ann_ret = np.exp(ret.mean() * TRADING_DAYS) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > EPS else np.nan
    dd = calculate_drawdown(ret)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > EPS else np.nan

    m = {"total_ret": total_ret, "ann_ret": ann_ret, "ann_vol": ann_vol,
         "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}

    if benchmark is not None:
        bmk = benchmark.copy()
        if isinstance(bmk, pd.DataFrame):
            bmk = bmk.iloc[:, 0]
        r_al, b_al = ret.align(bmk.dropna(), join="inner")
        if len(r_al) >= 2:
            active = r_al - b_al
            te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
            ir = (np.exp(r_al.mean() * TRADING_DAYS) - 1.0 - np.exp(b_al.mean() * TRADING_DAYS) + 1.0) / te if te > EPS else np.nan
            m["tracking_error"] = te
            m["info_ratio"] = ir

    return m


# ═════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN A EXCEL — formato Coril SAB
# ═════════════════════════════════════════════════════════════════════════════

def export_portfolio_xlsx(
    weights, profile_name, ret_bl, sigma_bl, asset_metadata,
    hist_returns, benchmark_returns, fico_forced=None,
    punto_idx=None,
):
    """
    Genera un archivo Excel en memoria (BytesIO) con formato Coril SAB.
    Hojas: Métricas, Composición, Wealth Index, Retornos Diarios,
           Circular Sector/Región/Instrumento/Clase.
    Retorna bytes listos para st.download_button.
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import LineChart, Reference
    import io

    if fico_forced is None:
        fico_forced = FICO_FORCED
    fico_ret = list(fico_forced.values())[0]["ret_annual"]
    fico_assets_list = list(fico_forced.keys())

    # Normalizar pesos
    w = weights.copy()
    if w.sum() > EPS:
        w = w / w.sum()

    eq_w = float(w[~w.index.isin(fico_assets_list)].sum())
    fi_w = float(w[w.index.isin(fico_assets_list)].sum())

    # Estilos
    BS = Side(style="thin", color="BFBFBF")
    CB = Border(left=BS, right=BS, top=BS, bottom=BS)
    FB = Font(name="Arial", size=10)
    AC = Alignment(horizontal="center", vertical="center")
    AL = Alignment(horizontal="left", vertical="center")
    AR = Alignment(horizontal="right", vertical="center")
    HX = "1F4E79"
    RF_FILL = "DEEAF1"

    def hf(c): return PatternFill("solid", fgColor=c)
    def rf(): return PatternFill("solid", fgColor=RF_FILL)

    def style_header(ws, row):
        fn = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        for cell in ws[row]:
            if cell.value is None: continue
            cell.fill = hf(HX); cell.font = fn; cell.alignment = AC; cell.border = CB

    def autofit(ws, pad=4, mx=40):
        for col in ws.columns:
            mw = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(mw+pad, mx)

    # Datos comunes
    common_idx = hist_returns.index.intersection(benchmark_returns.index)
    hist = hist_returns.loc[common_idx]
    bmk = benchmark_returns.loc[common_idx]

    # Métricas
    ann_vol_p = float(hist.std(ddof=1) * np.sqrt(TRADING_DAYS))
    max_dd_p = calculate_max_drawdown(hist)
    ann_ret_p = float(sum(w.get(a, 0) * ret_bl.loc[a] for a in w.index if a in ret_bl.index))
    contrib_eq = sum(w.get(a, 0) * ret_bl.loc[a] for a in w.index
                     if a in ret_bl.index and a not in fico_assets_list)
    contrib_fi = sum(w.get(a, 0) * ret_bl.loc[a] for a in w.index
                     if a in ret_bl.index and a in fico_assets_list)
    sharpe_p = (ann_ret_p - RF_ANNUAL) / ann_vol_p if ann_vol_p > EPS else np.nan
    ann_ret_b = float(np.exp(bmk.mean() * TRADING_DAYS) - 1)
    ann_vol_b = float(bmk.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_b = (ann_ret_b - RF_ANNUAL) / ann_vol_b if ann_vol_b > EPS else np.nan
    max_dd_b = calculate_max_drawdown(bmk)
    beta_vec = pd.to_numeric(asset_metadata.reindex(w.index)["beta"], errors="coerce").fillna(1.0)
    beta_port = float((w * beta_vec).sum())

    label = f"{profile_name}"
    if punto_idx is not None:
        label += f" | Punto {punto_idx}"

    wb = Workbook()
    wb.remove(wb.active)

    # ═══ Hoja 1: Métricas ═══
    ws = wb.create_sheet("Métricas")
    ws.merge_cells("A1:E1")
    ws["A1"].value = f"CORIL SAB — {label} | RV={eq_w:.0%} / RF={fi_w:.0%}"
    ws["A1"].font = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    ws["A1"].fill = hf(HX); ws["A1"].alignment = AC
    ws.row_dimensions[1].height = 22

    ws.merge_cells("A2:B2")
    ws["A2"].value = f"Renta Fija: {fi_w:.0%}  →  FICO {fico_ret:.2%}"
    ws["A2"].font = Font(name="Arial", bold=True, color="1F4E79", size=10)
    ws["A2"].fill = hf("DEEAF1"); ws["A2"].alignment = AC
    ws.merge_cells("C2:E2")
    ws["C2"].value = f"Equities: {eq_w:.0%}  →  retorno BL"
    ws["C2"].font = Font(name="Arial", bold=True, color="843C0C", size=10)
    ws["C2"].fill = hf("FCE4D6"); ws["C2"].alignment = AC

    ws.append([])
    ws.append(["RETORNO ESPERADO — DESGLOSE", "Portafolio", "", "", ""])
    style_header(ws, 4)
    for r in [
        (f"  Equity ({eq_w:.0%}) × BL", contrib_eq, "", "", ""),
        (f"  FICO   ({fi_w:.0%}) × {fico_ret:.2%}", contrib_fi, "", "", ""),
        ("  Retorno esperado total", ann_ret_p, "", "", ""),
    ]:
        ws.append(list(r))
    for ri, row in enumerate(ws.iter_rows(min_row=5, max_row=7), 5):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.border = CB
            cell.font = Font(name="Arial", bold=(ri == 7), size=10)
            cell.alignment = AL if ci == 1 else AR
            if ci == 2 and isinstance(cell.value, float): cell.number_format = "0.00%"

    ws.append([])
    bmk_name = benchmark_returns.name if hasattr(benchmark_returns, 'name') else "Benchmark"
    ws.append(["MÉTRICAS", "Portafolio", bmk_name, "Diferencia", ""])
    style_header(ws, 9)
    for r in [
        ("Retorno Esperado Total", ann_ret_p, ann_ret_b, ann_ret_p - ann_ret_b),
        ("Volatilidad Anual", ann_vol_p, ann_vol_b, ann_vol_p - ann_vol_b),
        (f"Sharpe Ratio (Rf={RF_ANNUAL:.2%})", sharpe_p, sharpe_b, sharpe_p - sharpe_b),
        ("Máximo Drawdown", max_dd_p, max_dd_b, max_dd_p - max_dd_b),
        ("Beta Portafolio", beta_port, 1.00, beta_port - 1.00),
        ("Tasa Libre de Riesgo (Rf)", RF_ANNUAL, RF_ANNUAL, 0.0),
        ("Retorno FICO asumido", fico_ret, np.nan, np.nan),
    ]:
        ws.append([r[0], r[1], r[2], r[3], ""])
    pct_rows = {10, 11, 12, 13, 16, 17}
    for ri, row in enumerate(ws.iter_rows(min_row=10, max_row=16), 10):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AL if ci == 1 else AR
            if ci in {2,3,4} and isinstance(cell.value, float):
                cell.number_format = "0.00%" if ri in pct_rows else "0.00"

    ws.append([])
    ws.append(["RENDIMIENTOS POR PERÍODO", "Portafolio", bmk_name, "Diferencia", ""])
    style_header(ws, 18)
    cy = hist.index.max().year
    ytd_p, ytd_b = hist[hist.index.year == cy], bmk[bmk.index.year == cy]
    for label_r, vp, vb in [
        ("YTD", float(np.exp(ytd_p.sum())-1), float(np.exp(ytd_b.sum())-1)),
        ("1 año", cumulative_return_for_period(hist, TRADING_DAYS), cumulative_return_for_period(bmk, TRADING_DAYS)),
        ("2 años", cumulative_return_for_period(hist, TRADING_DAYS*2), cumulative_return_for_period(bmk, TRADING_DAYS*2)),
        ("3 años", cumulative_return_for_period(hist, TRADING_DAYS*3), cumulative_return_for_period(bmk, TRADING_DAYS*3)),
    ]:
        diff = (vp-vb) if pd.notna(vp) and pd.notna(vb) else np.nan
        ws.append([label_r, vp, vb, diff, ""])
    for row in ws.iter_rows(min_row=19, max_row=22):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AL if ci == 1 else AR
            if ci in {2,3,4} and isinstance(cell.value, float): cell.number_format = "0.00%"
    autofit(ws); ws.freeze_panes = "A5"

    # ═══ Hoja 2: Composición ═══
    ws2 = wb.create_sheet("Composición")
    ws2.append(["Activo","Clase","Sector","Región","Instrumento","Peso","Beta","Vol (%)","Ret.BL","Contrib.Beta"])
    style_header(ws2, 1)
    for asset, peso in w.sort_values(ascending=False).items():
        if peso < EPS: continue
        clase = "RF" if asset in fico_assets_list else "EQ"
        meta = asset_metadata.loc[asset] if asset in asset_metadata.index else {}
        def g(d,k):
            v = d.get(k,"N/A") if isinstance(d,dict) else (d[k] if k in d.index else "N/A")
            return v if isinstance(v,str) else str(v)
        ba = pd.to_numeric(asset_metadata.loc[asset,"beta"] if asset in asset_metadata.index else np.nan, errors="coerce")
        va = float(np.sqrt(sigma_bl.loc[asset,asset])) if asset in sigma_bl.index else np.nan
        ra = ret_bl.loc[asset] if asset in ret_bl.index else np.nan
        ct = peso*ba if pd.notna(ba) else np.nan
        ws2.append([asset, clase, g(meta,"sector"), g(meta,"region"), g(meta,"instrumento"),
                    round(peso,6), round(float(ba),4) if pd.notna(ba) else None,
                    round(float(va)*100,2) if pd.notna(va) else None,
                    round(float(ra),6) if pd.notna(ra) else None,
                    round(float(ct),6) if pd.notna(ct) else None])
    n2 = w[w>=EPS].shape[0]
    ws2.append(["TOTAL","","","","",f"=SUM(F2:F{1+n2})","","","",f"=SUM(J2:J{1+n2})"])
    for row in ws2.iter_rows(min_row=2, max_row=n2+2):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AL if ci<=5 else AR
            if ci==6 and isinstance(cell.value,float): cell.number_format = "0.00%"
            if ci==7 and isinstance(cell.value,float): cell.number_format = "0.000"
            if ci==8 and isinstance(cell.value,float): cell.number_format = "0.00"
            if ci in(9,10) and isinstance(cell.value,float): cell.number_format = "0.00%"
    for cell in ws2[n2+2]: cell.font = Font(name="Arial",bold=True,size=10)
    autofit(ws2); ws2.freeze_panes = "A2"

    # ═══ Hoja 3: Wealth Index ═══
    ws3 = wb.create_sheet("Wealth Index")
    ws3.append(["Fecha", profile_name, bmk_name]); style_header(ws3, 1)
    wp = np.exp(hist.cumsum())*100; wb_idx = np.exp(bmk.cumsum())*100
    for dt, vp, vb in zip(hist.index.strftime("%Y-%m-%d"), wp.round(4).values, wb_idx.round(4).values):
        ws3.append([dt, float(vp), float(vb)])
    for row in ws3.iter_rows(min_row=2):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AR if ci>1 else AL
            if ci in(2,3) and isinstance(cell.value,float): cell.number_format = "0.00"
    autofit(ws3); ws3.freeze_panes = "A2"
    n_wi = len(hist)

    # ═══ Hoja 4: Retornos Diarios ═══
    ws4 = wb.create_sheet("Retornos Diarios")
    ws4.append(["Fecha", profile_name, bmk_name, "Drawdown Port.", "Drawdown Bench."])
    style_header(ws4, 1)
    dd_p = calculate_drawdown(hist); dd_b = calculate_drawdown(bmk)
    for dt, rp, rb, dp, db in zip(hist.index.strftime("%Y-%m-%d"),
        hist.values, bmk.values, dd_p.values, dd_b.values):
        ws4.append([dt, round(float(rp),6), round(float(rb),6), round(float(dp),6), round(float(db),6)])
    for row in ws4.iter_rows(min_row=2):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AR if ci>1 else AL
            if ci in(2,3) and isinstance(cell.value,float): cell.number_format = "0.0000%"
            if ci in(4,5) and isinstance(cell.value,float): cell.number_format = "0.00%"
    autofit(ws4); ws4.freeze_panes = "A2"

    # ═══ Hojas Circulares ═══
    for sn, mc, al in [("Circular Sector","sector","Sector"),
                       ("Circular Región","region","Región"),
                       ("Circular Instrumento","instrumento","Instrumento")]:
        wsX = wb.create_sheet(sn)
        wsX.append([al, "Peso"]); style_header(wsX, 1)
        agg = {}
        for asset, peso in w.items():
            if peso < EPS: continue
            val = asset_metadata.loc[asset, mc] if asset in asset_metadata.index else "N/A"
            val = val if isinstance(val, str) else str(val)
            agg[val] = agg.get(val, 0) + peso
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            wsX.append([k, round(v, 6)])
        for row in wsX.iter_rows(min_row=2):
            for ci, cell in enumerate(row, 1):
                cell.fill = rf(); cell.font = FB; cell.border = CB
                cell.alignment = AL if ci==1 else AR
                if ci==2 and isinstance(cell.value, float): cell.number_format = "0.00%"
        autofit(wsX)

    ws8 = wb.create_sheet("Circular Clase")
    ws8.append(["Clase","Peso"]); style_header(ws8, 1)
    for k, v in [("Renta Variable (EQ)", eq_w), ("Renta Fija (RF)", fi_w)]:
        ws8.append([k, round(v, 6)])
    for row in ws8.iter_rows(min_row=2):
        for ci, cell in enumerate(row, 1):
            cell.fill = rf(); cell.font = FB; cell.border = CB
            cell.alignment = AL if ci==1 else AR
            if ci==2 and isinstance(cell.value, float): cell.number_format = "0.00%"
    autofit(ws8)

    # ═══ Gráfico Wealth ═══
    ws9 = wb.create_sheet("Gráfico Wealth")
    chart = LineChart()
    chart.title = f"Wealth Index — {profile_name} vs {bmk_name} (Base 100)"
    chart.style = 10; chart.y_axis.title = "Índice (Base 100)"
    chart.x_axis.title = "Fecha"; chart.height = 14; chart.width = 26
    chart.add_data(Reference(ws3, min_col=2, min_row=1, max_row=n_wi+1), titles_from_data=True)
    chart.add_data(Reference(ws3, min_col=3, min_row=1, max_row=n_wi+1), titles_from_data=True)
    chart.set_categories(Reference(ws3, min_col=1, min_row=2, max_row=n_wi+1))
    chart.series[0].graphicalProperties.line.solidFill = "1F77B4"
    chart.series[0].graphicalProperties.line.width = 20000
    chart.series[1].graphicalProperties.line.solidFill = "808080"
    chart.series[1].graphicalProperties.line.width = 15000
    ws9.add_chart(chart, "A1")

    # Guardar en memoria
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()
