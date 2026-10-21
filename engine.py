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

# ── Layout Excel ──────────────────────────────────────────────────────────────
REGION_ROW     = 0
SECTOR_ROW     = 1
CURRENCY_ROW   = 2
INSTRUMENT_ROW = 3
BETA_ROW       = 4
TICKER_ROW     = 5
DATA_START_ROW = 6
DATE_COL_CANDIDATES = ["Date", "Fecha", "date", "fecha"]

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
# CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def _clean_label(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in ("nan", "none", "") else value


def _remove_timezone(index):
    index = pd.to_datetime(index)
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _detect_date_column(df):
    for col in DATE_COL_CANDIDATES:
        if col in df.columns:
            return col
    first_col = df.columns[0]
    if pd.to_datetime(df[first_col], errors="coerce").notna().mean() >= 0.80:
        return first_col
    raise ValueError(f"No se detectó columna de fechas: {df.columns.tolist()}")


def load_single_sheet(file_path, sheet_name):
    """Carga una hoja de Excel con el layout Coril (metadata en filas 0-5, datos desde 6)."""
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    if raw.shape[0] <= DATA_START_ROW:
        raise ValueError(f"Hoja '{sheet_name}': solo {raw.shape[0]} filas.")

    region_vals     = raw.iloc[REGION_ROW].map(_clean_label)
    sector_vals     = raw.iloc[SECTOR_ROW].map(_clean_label)
    currency_vals   = raw.iloc[CURRENCY_ROW].map(_clean_label)
    instrument_vals = raw.iloc[INSTRUMENT_ROW].map(_clean_label)
    beta_raw        = raw.iloc[BETA_ROW]
    ticker_vals     = raw.iloc[TICKER_ROW].map(_clean_label)

    valid_cols = [i for i, t in enumerate(ticker_vals) if t and not t.startswith("Unnamed")]
    if not valid_cols:
        raise ValueError(f"Sin columnas válidas en '{sheet_name}'.")

    columns = [ticker_vals.iloc[i].strip() for i in valid_cols]
    data = raw.iloc[DATA_START_ROW:, valid_cols].copy()
    data.columns = columns

    dcol = _detect_date_column(data)
    data[dcol] = pd.to_datetime(data[dcol], errors="coerce")
    data = data.dropna(subset=[dcol]).set_index(dcol)
    data.index = _remove_timezone(data.index)
    data = data[~data.index.duplicated(keep="last")].sort_index()

    candidates = [c for c in data.columns if c.strip() and not c.startswith("Unnamed") and c not in DATE_COL_CANDIDATES]
    prices = data[candidates].apply(pd.to_numeric, errors="coerce")
    prices = prices.replace(0, np.nan)
    prices[prices < 0] = np.nan
    prices = prices.dropna(axis=1, how="all")

    miss_pct = prices.isna().mean()
    valid = [a for a in miss_pct.index
             if miss_pct[a] <= (MAX_MISSING_PCT_LATE if a in LATE_START_ASSETS else MAX_MISSING_PCT)]
    prices = prices[valid].ffill(limit=MAX_STALE_DAYS).dropna(how="all")

    meta_rows = []
    for i, asset in zip(valid_cols, columns):
        if asset not in prices.columns or asset in DATE_COL_CANDIDATES:
            continue
        meta_rows.append({
            "asset": asset,
            "region": region_vals.iloc[i] or "Sin clasificar",
            "sector": sector_vals.iloc[i] or "Sin clasificar",
            "moneda": currency_vals.iloc[i] or "Sin clasificar",
            "instrumento": instrument_vals.iloc[i] or "Sin clasificar",
            "beta": pd.to_numeric(beta_raw.iloc[i], errors="coerce"),
            "asset_class": "Renta Variable",
            "fuente": "Excel",
        })

    metadata = pd.DataFrame(meta_rows).set_index("asset").reindex(prices.columns)
    return prices, metadata


def calculate_returns(price_data):
    prices = price_data.copy().apply(pd.to_numeric, errors="coerce")
    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if WINSORIZE_LIMIT and WINSORIZE_LIMIT > 0:
        lower = log_ret.quantile(WINSORIZE_LIMIT, axis=0)
        upper = log_ret.quantile(1.0 - WINSORIZE_LIMIT, axis=0)
        log_ret = log_ret.clip(lower=lower, upper=upper, axis=1)

    col_nan = log_ret.isna().mean()
    bad = [c for c in col_nan.index
           if col_nan[c] > (MAX_MISSING_PCT_LATE if c in LATE_START_ASSETS else MAX_MISSING_PCT)]
    if bad:
        log_ret = log_ret.drop(columns=bad)
    return log_ret.dropna(how="all").fillna(0.0)


def build_synthetic_fico(fico_forced, date_index, seed=42):
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


def load_full_universe(excel_file, sheet_names, fico_forced=None):
    """Pipeline completo: Excel → returns → FICO → benchmark → alineación."""
    if fico_forced is None:
        fico_forced = FICO_FORCED

    # 1. Equity
    all_prices, all_meta = [], []
    for sheet in sheet_names:
        p, m = load_single_sheet(excel_file, sheet)
        all_prices.append(p)
        all_meta.append(m)
    eq_prices = pd.concat(all_prices, axis=1, join="outer").sort_index()
    eq_prices = eq_prices.ffill(limit=MAX_STALE_DAYS).dropna(how="all")
    eq_meta = pd.concat(all_meta, axis=0)
    eq_prices = eq_prices.loc[:, ~eq_prices.columns.duplicated()]
    eq_meta = eq_meta[~eq_meta.index.duplicated(keep="first")]

    # 2. Returns
    rets_eq = calculate_returns(eq_prices)

    # 3. FICO
    rets_fi, fi_meta = build_synthetic_fico(fico_forced, rets_eq.index)

    # 4. Consolidar
    returns = pd.concat([rets_eq, rets_fi], axis=1, join="outer")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)

    equity_assets = [a for a in eq_prices.columns if a in returns.columns]
    fico_assets = list(fico_forced.keys())
    asset_metadata = pd.concat([eq_meta, fi_meta], axis=0).reindex(returns.columns)

    # 5. Benchmark
    idx = _remove_timezone(pd.to_datetime(returns.index))
    start = (idx.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (idx.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    bmk_ret = download_benchmark(start=start, end=end)

    # 6. Alinear
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
