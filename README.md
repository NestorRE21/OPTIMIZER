# Coril SAB — Optimizador de Portafolios Black-Litterman

Aplicación de optimización de portafolios basada en el modelo **Black-Litterman** con restricciones prácticas de inversión.

## Características

- **Black-Litterman completo**: retornos implícitos de equilibrio (reverse optimization), actualización bayesiana con views del analista, matriz Omega escalada (He-Litterman)
- **Frontera eficiente por perfil**: Conservador (30/70), Moderado (50/50), Agresivo (70/30)
- **5 perfiles con activos fijos**: Perfil A (30/70) a Perfil E (70/30)
- **Restricciones**: buckets RV/RF, beta, sector, max weight, drawdown
- **Covarianza robusta**: Ledoit-Wolf shrinkage + regularización espectral (Higham 2002)
- **Backtest histórico**: curvas de riqueza, drawdown, Sharpe, tracking error vs benchmark
- **Portafolio manual**: análisis de pesos personalizados

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Cloud

1. Push a GitHub
2. En [share.streamlit.io](https://share.streamlit.io), conectar el repo
3. Main file: `app.py`
4. **No usar `runtime.txt`** — causa problemas de dependencias

## Estructura

```
├── app.py              # Streamlit UI
├── config.py           # Constantes y perfiles
├── core/
│   ├── __init__.py
│   └── engine.py       # Motor BL + optimización + métricas
├── requirements.txt
└── README.md
```

## Uso

1. Sube el archivo `AF.xlsx` (renta variable, formato Coril)
2. Configura FICO, τ, Rf en el sidebar
3. Define views del analista (pestaña Views)
4. Ejecuta la optimización
5. Revisa backtest y ajusta

---

*Grupo Coril SAB — Herramienta interna de análisis. No constituye recomendación de inversión.*
