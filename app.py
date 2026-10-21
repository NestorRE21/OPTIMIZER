"""
CORIL SAB — Optimizador de Portafolios Black-Litterman
Streamlit App
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Coril SAB — Portafolio BL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from engine import (
    # Config constants
    TRADING_DAYS, RF_ANNUAL, EPS, MIN_EQUITY_ASSETS, MAX_EQUITY_ASSETS,
    MIN_FICO_ASSETS, MAX_FICO_ASSETS, MAX_CONSERVATIVE_VOL_ANNUAL,
    PROFILE_TARGETS, PROFILE_MAX_DRAWDOWN, PROFILE_MIN_BETA, PROFILE_MAX_BETA,
    PROFILE_EXCLUSIONS, FIXED_PROFILES, FIVE_PROFILE_BETA_MAP,
    FICO_USD, FICO_PEN, FICO_FORCED,
    # Functions
    load_full_universe, get_market_cap_weights, calibrate_market,
    build_views, black_litterman_posterior, select_equity_for_profile,
    optimize_frontier, pick_portfolio, performance_metrics,
    calculate_drawdown, cumulative_return_for_period,
    portfolio_historical_returns, sector_exposure_from_weights,
    calculate_max_drawdown, nearest_psd,
)

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "bl_computed" not in st.session_state:
    st.session_state.bl_computed = False


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Configuración")

    st.subheader("📁 Datos")
    uploaded_file = st.file_uploader("Sube AF.xlsx (renta variable)", type=["xlsx"])
    sheet_names = st.text_input("Hojas (separadas por coma)", value="USD").split(",")
    sheet_names = [s.strip() for s in sheet_names if s.strip()]

    st.subheader("🏦 FICO")
    fico_currency = st.selectbox("Moneda FICO", ["USD", "PEN"])
    fico_forced = FICO_USD if fico_currency == "USD" else FICO_PEN
    fico_ticker = list(fico_forced.keys())[0]
    fico_ret = st.number_input(f"Retorno anual {fico_ticker}", value=fico_forced[fico_ticker]["ret_annual"],
                                min_value=-0.50, max_value=2.0, step=0.005, format="%.3f")
    fico_forced[fico_ticker]["ret_annual"] = fico_ret

    st.subheader("📐 Parámetros BL")
    tau = st.slider("τ (incertidumbre equilibrio)", 0.01, 0.30, 0.05, 0.01)
    risk_free = st.number_input("Rf anual", value=RF_ANNUAL, step=0.005, format="%.3f")

    st.subheader("🎯 Modo")
    mode = st.radio("Pipeline", ["3 Perfiles (screening)", "5 Perfiles (activos fijos)"])

    if mode == "5 Perfiles (activos fijos)":
        st.subheader("📌 Activos fijos (RV)")
        # Se definen después de cargar datos
        if st.session_state.data_loaded:
            all_eq = st.session_state.equity_assets
            default_fixed = [a for a in ["SPY PE Equity", "GLD PE Equity", "GOOGL PE Equity", "BAP PE Equity"] if a in all_eq]
            fixed_equity = st.multiselect("Seleccionar activos RV", all_eq, default=default_fixed)
            st.session_state.fixed_equity = fixed_equity

    run_btn = st.button("🚀 Ejecutar pipeline", type="primary", use_container_width=True)
    if st.session_state.data_loaded:
        st.success(f"✅ {st.session_state.n_assets} activos cargados")


# ═════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def cached_load(file_bytes, sheets, _fico):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        return load_full_universe(tmp, sheets, _fico)
    finally:
        os.unlink(tmp)


if uploaded_file is not None and run_btn:
    with st.spinner("📊 Cargando universo + benchmark..."):
        try:
            returns, eq_assets, fico_assets, metadata, bmk = cached_load(
                uploaded_file.getvalue(), sheet_names, fico_forced
            )
            # Guardar en session_state
            st.session_state.returns = returns
            st.session_state.equity_assets = eq_assets
            st.session_state.fico_assets = fico_assets
            st.session_state.metadata = metadata
            st.session_state.benchmark = bmk
            st.session_state.n_assets = len(returns.columns)
            st.session_state.data_loaded = True

            # Calibrar
            mkt_params = calibrate_market(returns, bmk, tau_method=tau)
            st.session_state.tau = mkt_params["tau"]
            st.session_state.lambda_mkt = mkt_params["lambda"]
            st.session_state.mkt_params = mkt_params

            st.success(f"✅ {len(returns.columns)} activos | {len(returns)} días | "
                       f"τ={mkt_params['tau']:.4f} | λ={mkt_params['lambda']:.4f}")
        except Exception as e:
            st.error(f"❌ Error cargando datos: {e}")
            st.stop()

elif not st.session_state.data_loaded:
    st.title("📊 Coril SAB — Optimizador BL")
    st.info("👈 Sube el archivo Excel y presiona **Ejecutar pipeline** para comenzar.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ═════════════════════════════════════════════════════════════════════════════

returns   = st.session_state.returns
eq_assets = st.session_state.equity_assets
fi_assets = st.session_state.fico_assets
metadata  = st.session_state.metadata
bmk       = st.session_state.benchmark
ASSETS    = returns.columns.tolist()

tab_universe, tab_views, tab_optimize, tab_backtest, tab_manual = st.tabs([
    "🌐 Universo", "👁️ Views", "📈 Optimización", "📉 Backtest", "✏️ Manual"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: UNIVERSO
# ═════════════════════════════════════════════════════════════════════════════

with tab_universe:
    st.header("Universo de inversión")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Activos", len(ASSETS))
    col2.metric("RV", len(eq_assets))
    col3.metric("RF", len(fi_assets))
    col4.metric("Días", len(returns))

    st.subheader("Metadata")
    st.dataframe(metadata[["sector", "region", "moneda", "instrumento", "beta", "asset_class"]].fillna("N/A"), use_container_width=True)

    st.subheader("Parámetros de mercado")
    mp = st.session_state.mkt_params
    cols = st.columns(4)
    cols[0].metric("τ", f"{mp['tau']:.4f}")
    cols[1].metric("λ", f"{mp['lambda']:.4f}")
    cols[2].metric("Sharpe BMK", f"{mp['sharpe_bmk']:.3f}")
    cols[3].metric("Ret. BMK", f"{mp['bmk_ret_ann']:.2%}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: VIEWS
# ═════════════════════════════════════════════════════════════════════════════

with tab_views:
    st.header("Views del analista")
    st.caption("Q = retorno total anual. Para views absolutas: Q_excess = Q − Rf automáticamente.")

    if "views_config" not in st.session_state:
        st.session_state.views_config = []

    n_views = st.number_input("Número de views", min_value=0, max_value=20, value=len(st.session_state.views_config), key="n_views_input")

    views_list = []
    for i in range(int(n_views)):
        with st.expander(f"View {i+1}", expanded=(i == 0)):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
            vtype = c1.selectbox("Tipo", ["absolute", "relative"], key=f"vtype_{i}")
            if vtype == "absolute":
                asset = c2.selectbox("Activo", eq_assets, key=f"vasset_{i}")
            else:
                asset = None
            q = c3.number_input("Q (anual)", value=0.10, step=0.01, format="%.2f", key=f"vq_{i}")
            c_val = c4.number_input("c", value=1.0, step=0.1, format="%.1f", key=f"vc_{i}")
            k_val = c5.number_input("k", value=1.0, step=0.1, format="%.1f", key=f"vk_{i}")

            view = {"type": vtype, "q": q, "c": c_val, "k": k_val, "name": f"View_{i+1}"}
            if vtype == "absolute":
                view["asset"] = asset
            elif vtype == "relative":
                long_a = st.selectbox("Long", eq_assets, key=f"vlong_{i}")
                short_a = st.selectbox("Short", [a for a in eq_assets if a != long_a], key=f"vshort_{i}")
                view["long"], view["short"] = long_a, short_a
            views_list.append(view)

    st.session_state.views_config = views_list

    if views_list:
        df_views = pd.DataFrame([{
            "View": v["name"], "Tipo": v["type"], "Q": f"{v['q']:.2%}",
            "c": v["c"], "k": v["k"], "k/c": f"{v['k']/v['c']:.2f}",
        } for v in views_list])
        st.dataframe(df_views, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: OPTIMIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════

with tab_optimize:
    st.header("Optimización por perfil")

    run_opt = st.button("🔄 Ejecutar BL + Optimización", type="primary", key="run_opt")

    if run_opt or st.session_state.bl_computed:
        if run_opt:
            with st.spinner("⏳ Ejecutando Black-Litterman..."):
                views_cfg = st.session_state.views_config
                mkt_target = PROFILE_TARGETS["Moderado"]
                mkt_w = get_market_cap_weights(ASSETS, eq_assets, fi_assets,
                                                mkt_target["equity"], mkt_target["fico"])

                if views_cfg:
                    P, Q, c_vec, k_vec, vnames = build_views(ASSETS, views_cfg)
                    bl = black_litterman_posterior(returns, P, Q, tau=st.session_state.tau,
                                                   risk_aversion=st.session_state.lambda_mkt,
                                                   market_weights=mkt_w.values, c=c_vec, k=k_vec)
                else:
                    n_a = len(ASSETS)
                    P_dummy = np.zeros((1, n_a))
                    P_dummy[0, 0] = 1.0
                    mu_dummy = returns.iloc[:, 0].mean() * TRADING_DAYS + risk_free
                    bl = black_litterman_posterior(returns, P_dummy, np.array([mu_dummy]),
                                                   tau=st.session_state.tau,
                                                   risk_aversion=st.session_state.lambda_mkt,
                                                   market_weights=mkt_w.values)

                st.session_state.bl = bl
                st.session_state.ret_bl = bl["ret_bl"]
                st.session_state.sigma_bl = bl["sigma_bl"]

            # Optimizar por perfil
            profiles_to_run = FIXED_PROFILES if mode == "5 Perfiles (activos fijos)" else PROFILE_TARGETS
            beta_map = FIVE_PROFILE_BETA_MAP if mode == "5 Perfiles (activos fijos)" else {k: k for k in PROFILE_TARGETS}

            results = {}
            ret_bl = st.session_state.ret_bl
            sigma_bl = st.session_state.sigma_bl

            for pname, targets in profiles_to_run.items():
                eq_t, fi_t = targets["equity"], targets["fico"]
                beta_profile = beta_map.get(pname, "Moderado")

                with st.spinner(f"📊 {pname}..."):
                    if mode == "5 Perfiles (activos fijos)":
                        sel_eq = st.session_state.get("fixed_equity", eq_assets[:4])
                    else:
                        sel_eq = select_equity_for_profile(beta_profile, ret_bl, sigma_bl, eq_assets, metadata)

                    sel_fico = fi_assets
                    frontier_assets = sel_eq + sel_fico

                    try:
                        frontier, weights_f = optimize_frontier(
                            ret_bl, sigma_bl, frontier_assets, sel_eq, sel_fico,
                            eq_t, fi_t, beta_profile, asset_metadata=metadata,
                            num_puntos=100)

                        result = pick_portfolio(beta_profile, frontier, weights_f,
                                                frontier_assets, sel_eq, sel_fico,
                                                eq_t, fi_t, returns, metadata)
                        result["profile_display"] = pname
                        result["selected_equity"] = sel_eq
                        results[pname] = result
                    except Exception as e:
                        st.warning(f"⚠️ {pname}: {e}")

            st.session_state.results = results
            st.session_state.bl_computed = True

        # ── Mostrar resultados
        results = st.session_state.get("results", {})
        if results:
            # Tabla comparativa
            st.subheader("Comparativa de perfiles")
            comp_data = []
            for pname, r in results.items():
                comp_data.append({
                    "Perfil": pname,
                    "Ret. BL": f"{r['return']:.2%}",
                    "Vol.": f"{r['risk']:.2%}",
                    "Sharpe": f"{r['sharpe']:.3f}",
                    "Max DD": f"{r['max_drawdown']:.2%}",
                    "RV": f"{r['equity_weight']:.2%}",
                    "RF": f"{r['fico_weight']:.2%}",
                    "Beta": f"{r['beta']:.3f}",
                })
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            # Fronteras
            st.subheader("Fronteras eficientes")
            cols = st.columns(min(len(results), 3))
            colors = {"Conservador": "#1F77B4", "Moderado": "#FF7F0E", "Agresivo": "#D62728",
                      "Perfil A (30/70)": "#9467BD", "Perfil B (40/60)": "#1F77B4",
                      "Perfil C (50/50)": "#2CA02C", "Perfil D (60/40)": "#FF7F0E",
                      "Perfil E (70/30)": "#D62728"}

            for i, (pname, r) in enumerate(results.items()):
                with cols[i % len(cols)]:
                    f = r["frontier"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=f[:, 1]*100, y=f[:, 0]*100,
                                             mode="lines", name="Frontera", line=dict(color="black", width=2)))
                    fig.add_trace(go.Scatter(x=[r["risk"]*100], y=[r["return"]*100],
                                             mode="markers", name=pname,
                                             marker=dict(size=12, color=colors.get(pname, "gray"))))
                    fig.update_layout(title=pname, xaxis_title="Vol (%)", yaxis_title="Ret (%)",
                                      height=350, margin=dict(t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)

            # Pesos por perfil
            st.subheader("Pesos por perfil")
            for pname, r in results.items():
                with st.expander(f"📋 {pname}"):
                    w = r["weights"]
                    df_w = pd.DataFrame({
                        "Activo": w.index,
                        "Peso": w.values,
                        "Clase": ["RV" if a in r.get("selected_equity", eq_assets) else "RF" for a in w.index],
                    })
                    df_w["Peso%"] = df_w["Peso"].map("{:.2%}".format)
                    st.dataframe(df_w[df_w["Peso"] > 0.001][["Activo", "Clase", "Peso%"]],
                                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.header("Backtest histórico")

    results = st.session_state.get("results", {})
    if not results:
        st.info("Ejecuta la optimización primero.")
    else:
        returns_bt = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        common = bmk.index
        port_rets = {}
        for pname, r in results.items():
            w = r["weights"].reindex(returns_bt.columns).fillna(0.0)
            w = w / w.sum()
            port_rets[pname] = returns_bt @ w
            common = common.intersection(port_rets[pname].dropna().index)

        bmk_bt = bmk.loc[common]
        colors_map = {"Conservador": "#1F77B4", "Moderado": "#FF7F0E", "Agresivo": "#D62728",
                      "Perfil A (30/70)": "#9467BD", "Perfil B (40/60)": "#1F77B4",
                      "Perfil C (50/50)": "#2CA02C", "Perfil D (60/40)": "#FF7F0E",
                      "Perfil E (70/30)": "#D62728"}

        # Wealth curve
        fig = go.Figure()
        for pname, s in port_rets.items():
            w_curve = np.exp(s.loc[common].cumsum())
            fig.add_trace(go.Scatter(x=w_curve.index, y=w_curve, name=pname,
                                     line=dict(color=colors_map.get(pname, "gray"), width=2.5)))
        w_bmk = np.exp(bmk_bt.cumsum())
        fig.add_trace(go.Scatter(x=w_bmk.index, y=w_bmk, name="Benchmark",
                                 line=dict(color="black", dash="dash", width=2)))
        fig.update_layout(title="Curva de Riqueza (base=1)", height=500, yaxis_title="Riqueza")
        st.plotly_chart(fig, use_container_width=True)

        # Métricas
        st.subheader("Métricas históricas")
        rows = []
        for pname, s in port_rets.items():
            m = performance_metrics(s.loc[common], benchmark=bmk_bt)
            rows.append({"Perfil": pname, **{k: f"{v:.2%}" if "ret" in k or "vol" in k or k in ("max_dd", "tracking_error", "total_ret")
                                              else f"{v:.3f}" for k, v in m.items()}})
        m_bmk = performance_metrics(bmk_bt)
        rows.append({"Perfil": "Benchmark", **{k: f"{v:.2%}" if "ret" in k or "vol" in k or k in ("max_dd", "total_ret")
                                                 else f"{v:.3f}" for k, v in m_bmk.items()}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: PORTAFOLIO MANUAL
# ═════════════════════════════════════════════════════════════════════════════

with tab_manual:
    st.header("Portafolio manual")

    if not st.session_state.bl_computed:
        st.info("Ejecuta BL primero.")
    else:
        st.caption("Define pesos manuales. Deben sumar 1.0.")
        manual_assets = st.multiselect("Activos", ASSETS, default=ASSETS[:5], key="manual_assets")

        if manual_assets:
            cols_w = st.columns(min(len(manual_assets), 4))
            weights_dict = {}
            for i, a in enumerate(manual_assets):
                with cols_w[i % len(cols_w)]:
                    w = st.number_input(a.split()[0], 0.0, 1.0, value=round(1.0/len(manual_assets), 2),
                                        step=0.01, format="%.2f", key=f"mw_{a}")
                    weights_dict[a] = w

            total = sum(weights_dict.values())
            st.metric("Suma de pesos", f"{total:.4f}", delta=f"{total - 1.0:+.4f}" if abs(total-1) > 0.001 else None)

            if abs(total - 1.0) < 0.01 and st.button("📊 Analizar portafolio manual"):
                w_m = pd.Series(weights_dict)
                w_m = w_m / w_m.sum()

                ret_bl = st.session_state.ret_bl
                sigma_bl = st.session_state.sigma_bl

                ret_exp = float(sum(w_m.get(a, 0) * ret_bl.loc[a] for a in w_m.index if a in ret_bl.index))
                w_vec = w_m.reindex(ASSETS).fillna(0.0).to_numpy()
                cov_np = sigma_bl.loc[ASSETS, ASSETS].to_numpy(dtype=float)
                vol_exp = float(np.sqrt(max(w_vec.T @ cov_np @ w_vec, EPS)))

                c1, c2, c3 = st.columns(3)
                c1.metric("Ret. esperado", f"{ret_exp:.2%}")
                c2.metric("Volatilidad", f"{vol_exp:.2%}")
                c3.metric("Sharpe", f"{(ret_exp - risk_free) / vol_exp:.3f}")

                # Backtest
                returns_bt = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
                w_norm = w_m.reindex(returns_bt.columns).fillna(0.0)
                w_norm = w_norm / w_norm.sum()
                hist = returns_bt @ w_norm
                m = performance_metrics(hist, benchmark=bmk)

                st.subheader("Métricas históricas")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ret. hist.", f"{m['ann_ret']:.2%}")
                c2.metric("Vol. hist.", f"{m['ann_vol']:.2%}")
                c3.metric("Max DD", f"{m['max_dd']:.2%}")
                c4.metric("Sharpe hist.", f"{m['sharpe']:.3f}")
