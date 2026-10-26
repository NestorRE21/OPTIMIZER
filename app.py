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
    load_full_universe, read_excel_tickers, get_market_cap_weights, calibrate_market,
    build_views, black_litterman_posterior, select_equity_for_profile,
    optimize_frontier, pick_portfolio, performance_metrics,
    calculate_drawdown, cumulative_return_for_period,
    portfolio_historical_returns, sector_exposure_from_weights,
    calculate_max_drawdown, nearest_psd,
    export_portfolio_xlsx, project_to_horizon,
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

    st.subheader("📁 Universo (Excel)")
    uploaded_file = st.file_uploader(
        "Sube AF.xlsx", type=["xlsx"],
        help="Tickers + metadata (sector, región, beta) desde el Excel. Los precios se descargan de Yahoo."
    )
    sheet_name = st.text_input("Hoja", value="USD")
    period = st.selectbox("Historia de precios (Yahoo)", ["3y", "5y", "10y", "max"], index=1,
                          help="Cuántos años de precios descargar de Yahoo Finance")

    # ── Pre-leer tickers del Excel para el selector de activos ────────────
    selected_tickers = None
    if uploaded_file is not None:
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as _f:
                _f.write(uploaded_file.getvalue())
                _tmp = _f.name
            all_tickers, _ = read_excel_tickers(_tmp, sheet_name)
            os.unlink(_tmp)

            st.subheader("✅ Activos a incluir")
            selected_tickers = st.multiselect(
                "Selecciona qué activos entran a la optimización",
                options=all_tickers,
                default=all_tickers,
                help="Por defecto se incluyen todos. Deselecciona los que no quieras optimizar."
            )
            if not selected_tickers:
                st.warning("⚠️ Selecciona al menos 1 activo.")
        except Exception as e:
            st.caption(f"(No se pudieron leer tickers aún: {e})")

    st.subheader("⏱️ Horizonte de inversión")
    HORIZON_OPTIONS = {
        "1 mes":    1/12,
        "6 meses":  0.5,
        "1 año":    1.0,
        "2 años":   2.0,
        "3 años":   3.0,
        "4 años":   4.0,
        "5 años":   5.0,
        "10 años":  10.0,
    }
    horizon_label = st.selectbox("Horizonte", list(HORIZON_OPTIONS.keys()), index=2,
                                  help="Período de tenencia del portafolio")
    horizon_years = HORIZON_OPTIONS[horizon_label]

    st.subheader("🏦 FICO")
    fico_currency = st.selectbox("Moneda FICO", ["USD", "PEN"])
    fico_forced = FICO_USD.copy() if fico_currency == "USD" else FICO_PEN.copy()
    fico_ticker = list(fico_forced.keys())[0]
    fico_ret = st.number_input(f"Retorno anual {fico_ticker}", value=fico_forced[fico_ticker]["ret_annual"],
                                min_value=-0.50, max_value=2.0, step=0.005, format="%.3f")
    fico_forced[fico_ticker]["ret_annual"] = fico_ret

    st.subheader("📐 Parámetros BL")
    tau = st.slider("τ (incertidumbre equilibrio)", 0.01, 1.00, 0.05, 0.01,
                    help="0.01-0.10: estándar | >0.30: views dominan | 1.0: override total")
    risk_free = st.number_input("Rf anual", value=RF_ANNUAL, step=0.005, format="%.3f")

    st.subheader("🎯 Modo")
    mode = st.radio("Pipeline", ["3 Perfiles (screening)", "5 Perfiles (activos fijos)"])

    if mode == "5 Perfiles (activos fijos)" and st.session_state.data_loaded:
        st.subheader("📌 Activos fijos (RV)")
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

@st.cache_data(ttl=3600, show_spinner="Cargando Excel + precios de Yahoo...")
def cached_load(_file_bytes, _sheet, _period, _fico_dict, _selected):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        f.write(_file_bytes)
        tmp = f.name
    try:
        return load_full_universe(tmp, sheet_name=_sheet, fico_forced=_fico_dict,
                                  period=_period, selected_tickers=_selected)
    finally:
        os.unlink(tmp)


if uploaded_file is not None and run_btn:
    # Resetear estado previo — fuerza recálculo completo
    st.session_state.bl_computed = False
    st.session_state.pop("results", None)
    st.session_state.pop("ret_bl", None)
    st.session_state.pop("sigma_bl", None)
    cached_load.clear()

    with st.spinner("📊 Cargando Excel + descargando precios de Yahoo..."):
        try:
            sel_tuple = tuple(selected_tickers) if selected_tickers else None
            returns, eq_assets, fico_assets, metadata, bmk = cached_load(
                uploaded_file.getvalue(), sheet_name, period, fico_forced, sel_tuple
            )
            st.session_state.returns = returns
            st.session_state.equity_assets = eq_assets
            st.session_state.fico_assets = fico_assets
            st.session_state.metadata = metadata
            st.session_state.benchmark = bmk
            st.session_state.n_assets = len(returns.columns)
            st.session_state.data_loaded = True
            st.session_state.horizon_years = horizon_years
            st.session_state.horizon_label = horizon_label

            # Calibrar con tau actual del sidebar
            mkt_params = calibrate_market(returns, bmk, tau_method=tau)
            st.session_state.tau = mkt_params["tau"]
            st.session_state.lambda_mkt = mkt_params["lambda"]
            st.session_state.mkt_params = mkt_params

            st.success(f"✅ {len(returns.columns)} activos | {len(returns)} días | "
                       f"Horizonte: {horizon_label} | "
                       f"τ={mkt_params['tau']:.4f} | λ={mkt_params['lambda']:.4f}")
        except Exception as e:
            st.error(f"❌ Error cargando datos: {e}")
            st.stop()

elif not st.session_state.data_loaded:
    st.title("📊 Coril SAB — Optimizador BL")
    st.info("👈 Sube el Excel (AF.xlsx) y presiona **Ejecutar pipeline** para comenzar.")
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

tab_universe, tab_views, tab_optimize, tab_explorer, tab_backtest, tab_manual = st.tabs([
    "🌐 Universo", "👁️ Views", "📈 Optimización", "🔍 Explorador", "📉 Backtest", "✏️ Manual"
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
    cols[3].metric("Ret. BMK anual", f"{mp['bmk_ret_ann']:.2%}")

    # Proyección al horizonte
    h_y = st.session_state.get("horizon_years", 1.0)
    h_label = st.session_state.get("horizon_label", "1 año")
    rf_horizon = (1.0 + RF_ANNUAL) ** h_y - 1.0
    bmk_ret_horizon = (1.0 + mp["bmk_ret_ann"]) ** h_y - 1.0
    bmk_vol_horizon = mp["bmk_vol_ann"] * np.sqrt(h_y)

    st.subheader(f"Proyección al horizonte ({h_label})")
    cols2 = st.columns(4)
    cols2[0].metric("Rf horizonte", f"{rf_horizon:.2%}")
    cols2[1].metric("Ret. BMK horizonte", f"{bmk_ret_horizon:.2%}")
    cols2[2].metric("Vol. BMK horizonte", f"{bmk_vol_horizon:.2%}")
    cols2[3].metric("Horizonte", h_label)


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

        # ── Mostrar resultados (siempre desde session_state)
        results = st.session_state.get("results", {})
        ret_bl = st.session_state.get("ret_bl")
        sigma_bl = st.session_state.get("sigma_bl")
        if results and ret_bl is not None:
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

            # Pesos por perfil + descarga Excel
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

                    # Descarga Excel
                    returns_bt = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
                    w_norm = w.reindex(returns_bt.columns).fillna(0.0)
                    w_norm = w_norm / w_norm.sum() if w_norm.sum() > EPS else w_norm
                    hist_p = returns_bt @ w_norm

                    xlsx_bytes = export_portfolio_xlsx(
                        weights=w, profile_name=pname,
                        ret_bl=ret_bl, sigma_bl=sigma_bl,
                        asset_metadata=metadata,
                        hist_returns=hist_p, benchmark_returns=bmk,
                        horizon_years=st.session_state.get("horizon_years", 1.0),
                        horizon_label=st.session_state.get("horizon_label", "1 año"),
                    )
                    fname = f"coril_{pname.replace(' ', '_').replace('/', '-')}.xlsx"
                    st.download_button(
                        f"⬇️ Descargar {pname} (Excel)",
                        data=xlsx_bytes, file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_opt_{pname}",
                    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPLORADOR DE FRONTERA
# ═════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    st.header("🔍 Explorador de frontera eficiente")

    results = st.session_state.get("results", {})
    ret_bl = st.session_state.get("ret_bl")
    sigma_bl = st.session_state.get("sigma_bl")
    if not results or ret_bl is None:
        st.info("Ejecuta la optimización primero (pestaña Optimización).")
    else:

        profile_names = list(results.keys())
        selected_profile = st.selectbox("Selecciona perfil", profile_names, key="explorer_profile")

        r = results[selected_profile]
        frontier = r["frontier"]
        weights_f = r["weights_frontier"]
        f_assets = r["frontier_assets"]
        sel_eq = r.get("selected_equity", eq_assets)
        n_points = len(frontier)

        # ── Slider para seleccionar punto
        punto = st.slider(
            f"Punto en la frontera ({n_points} puntos)",
            min_value=0, max_value=n_points - 1,
            value=min(int(n_points * 0.75), n_points - 1),
            key="explorer_punto"
        )

        # ── Extraer pesos del punto seleccionado
        w_raw = weights_f[punto]
        w_point = pd.Series(w_raw, index=f_assets, name=selected_profile)
        w_point = w_point[w_point > 0.001]
        if w_point.sum() > EPS:
            w_point = w_point / w_point.sum()

        ret_p = float(frontier[punto, 0])
        vol_p = float(frontier[punto, 1])
        sharpe_p = (ret_p - RF_ANNUAL) / vol_p if vol_p > EPS else 0.0

        # ── Métricas del punto
        st.subheader(f"Punto {punto} — {selected_profile}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ret. esperado", f"{ret_p:.2%}")
        c2.metric("Volatilidad", f"{vol_p:.2%}")
        c3.metric("Sharpe", f"{sharpe_p:.3f}")

        # Beta
        beta_vec = pd.to_numeric(
            metadata.reindex(w_point.index)["beta"], errors="coerce"
        ).fillna(1.0)
        beta_p = float((w_point * beta_vec).sum())
        c4.metric("Beta", f"{beta_p:.3f}")

        # ── Histórico
        returns_bt = returns.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        w_norm = w_point.reindex(returns_bt.columns).fillna(0.0)
        if w_norm.sum() > EPS:
            w_norm = w_norm / w_norm.sum()
        hist_ret = returns_bt @ w_norm
        ann_ret_hist = float(np.exp(hist_ret.mean() * TRADING_DAYS) - 1.0)
        ann_vol_hist = float(hist_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        max_dd = calculate_max_drawdown(hist_ret)

        c5, c6, c7 = st.columns(3)
        c5.metric("Ret. hist. (anual)", f"{ann_ret_hist:.2%}")
        c6.metric("Vol. hist. (anual)", f"{ann_vol_hist:.2%}")
        c7.metric("Max Drawdown", f"{max_dd:.2%}")

        # Proyección al horizonte
        h_y = st.session_state.get("horizon_years", 1.0)
        h_label = st.session_state.get("horizon_label", "1 año")
        ret_horizon = (1.0 + ret_p) ** h_y - 1.0
        vol_horizon = vol_p * np.sqrt(h_y)
        rf_h = (1.0 + RF_ANNUAL) ** h_y - 1.0
        sharpe_h = (ret_horizon - rf_h) / vol_horizon if vol_horizon > EPS else 0.0

        st.caption(f"📐 Proyección al horizonte: **{h_label}**")
        h1, h2, h3 = st.columns(3)
        h1.metric(f"Ret. esperado ({h_label})", f"{ret_horizon:.2%}")
        h2.metric(f"Vol. ({h_label})", f"{vol_horizon:.2%}")
        h3.metric(f"Sharpe ({h_label})", f"{sharpe_h:.3f}")

        # ── Layout: stackplot + tabla lado a lado
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            # ── Stackplot con plotly
            st.subheader("Composición a lo largo de la frontera")

            asset_classes = {a: ("RV" if a in sel_eq else "RF") for a in f_assets}

            # Ordenar: RV primero, luego RF
            order = sorted(f_assets, key=lambda a: (0 if asset_classes[a] == "RV" else 1,
                                                      -weights_f[:, f_assets.index(a)].mean()))

            rv_palette = ["#E45756", "#F58518", "#EECA3B", "#54A24B"]
            rf_palette = ["#4C78A8", "#72B7B2", "#9ECAE9"]

            fig_stack = go.Figure()
            rv_i, rf_i = 0, 0
            for asset in order:
                idx_a = f_assets.index(asset)
                y_vals = weights_f[:, idx_a]
                if asset_classes[asset] == "RV":
                    color = rv_palette[rv_i % len(rv_palette)]
                    rv_i += 1
                else:
                    color = rf_palette[rf_i % len(rf_palette)]
                    rf_i += 1
                fig_stack.add_trace(go.Scatter(
                    x=list(range(n_points)), y=y_vals,
                    mode="lines", name=asset, stackgroup="one",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                ))

            # Líneas de GMVP y punto seleccionado
            idx_gmvp = int(np.argmin(frontier[:, 1]))
            fig_stack.add_vline(x=idx_gmvp, line_dash="dot", line_color="black",
                                annotation_text="GMVP", annotation_position="top")
            fig_stack.add_vline(x=punto, line_dash="dash", line_color="#FF4B4B",
                                line_width=2.5,
                                annotation_text=f"← Punto {punto}",
                                annotation_position="top right")

            fig_stack.update_layout(
                height=450,
                yaxis=dict(tickformat=".0%", title="Peso"),
                xaxis=dict(title="Punto en la frontera"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                            xanchor="center", x=0.5, font=dict(size=10)),
                margin=dict(t=30, b=80),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        with col_table:
            # ── Tabla de pesos
            st.subheader("Pesos del portafolio")

            rows_w = []
            for asset, peso in w_point.sort_values(ascending=False).items():
                clase = "RV" if asset in sel_eq else "RF"
                beta_a = metadata.loc[asset, "beta"] if asset in metadata.index else np.nan
                ret_a = ret_bl.loc[asset] if asset in ret_bl.index else np.nan
                rows_w.append({
                    "Activo": asset,
                    "Clase": clase,
                    "Peso": peso,
                    "Beta": beta_a,
                    "Ret.BL": ret_a,
                })

            df_weights = pd.DataFrame(rows_w)
            df_display = df_weights.copy()
            df_display["Peso"] = df_display["Peso"].map("{:.2%}".format)
            df_display["Beta"] = df_display["Beta"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            df_display["Ret.BL"] = df_display["Ret.BL"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # ── Exposición RV / RF
            eq_w = float(w_point[w_point.index.isin(sel_eq)].sum())
            fi_w = 1.0 - eq_w
            st.markdown(f"**RV:** {eq_w:.1%} &nbsp;|&nbsp; **RF:** {fi_w:.1%} &nbsp;|&nbsp; **Beta:** {beta_p:.3f}")

            # ── Exposición sectorial
            st.subheader("Sectores")
            sect = sector_exposure_from_weights(w_point, metadata)
            df_sect = pd.DataFrame({"Sector": sect.index, "Peso": sect.values})
            df_sect["Peso"] = df_sect["Peso"].map("{:.1%}".format)
            st.dataframe(df_sect, use_container_width=True, hide_index=True)

        # ── Descargar portafolio como Excel
        st.divider()
        st.subheader("📥 Descargar portafolio")

        xlsx_explorer = export_portfolio_xlsx(
            weights=w_point, profile_name=selected_profile,
            ret_bl=ret_bl, sigma_bl=sigma_bl,
            asset_metadata=metadata,
            hist_returns=hist_ret, benchmark_returns=bmk,
            punto_idx=punto,
            horizon_years=st.session_state.get("horizon_years", 1.0),
            horizon_label=st.session_state.get("horizon_label", "1 año"),
        )
        fname_exp = f"coril_{selected_profile.replace(' ','_')}_punto{punto}.xlsx"
        st.download_button(
            label=f"⬇️ Descargar punto {punto} — {selected_profile} (Excel)",
            data=xlsx_explorer, file_name=fname_exp,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key="dl_explorer",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: BACKTEST
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

    ret_bl_m = st.session_state.get("ret_bl")
    sigma_bl_m = st.session_state.get("sigma_bl")
    if not st.session_state.bl_computed or ret_bl_m is None:
        st.info("Ejecuta BL primero (pestaña Optimización).")
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

                ret_bl = ret_bl_m
                sigma_bl = sigma_bl_m

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

                # Descarga Excel
                st.divider()
                xlsx_manual = export_portfolio_xlsx(
                    weights=w_m, profile_name="Manual",
                    ret_bl=ret_bl, sigma_bl=sigma_bl,
                    asset_metadata=metadata,
                    hist_returns=hist, benchmark_returns=bmk,
                    horizon_years=st.session_state.get("horizon_years", 1.0),
                    horizon_label=st.session_state.get("horizon_label", "1 año"),
                )
                st.download_button(
                    "⬇️ Descargar portafolio manual (Excel)",
                    data=xlsx_manual,
                    file_name="coril_portafolio_manual.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_manual",
                )
