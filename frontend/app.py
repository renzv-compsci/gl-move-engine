import streamlit as st 
import yfinance as yf 
import requests
import pandas as pd 

TICKERS = ["SPY", "QQQ", "GLD", "0005.HK", "EPHE"]
API_URL = "http://localhost:8000/api/v1/surveillance/predict"

st.set_page_config(
    page_title="GL-Move Engine Terminal", 
    layout="wide"
)

st.title("GL-Move Engine: Market Surveillance Terminal")
st.markdown("Automated algorithmic regime detection and cross-asset portfolio routing engine.")
st.divider()

st.sidebar.header("System Controls")
st.sidebar.markdown("---")
st.sidebar.info("Connected Backend Target:\n`http://localhost:8000`")

fetch_trigger = st.button("Fetch Live Market Feed & Run Inference", use_container_width=True)

if fetch_trigger: 
    with st.spinner("Establishing secure handshake with Yahoo Finance data matrix."): 
        try: 
            raw_data = yf.download(TICKERS, period="45d")['Close']
            raw_data = raw_data.ffill().bfill().dropna()

            if len(raw_data) < 32: 
                st.error(f"Market feed returned insufficient timeframe ({len(raw_data)} rows). 32 consecutive days required.")
                st.stop()

            inference_subset = raw_data.tail(32)
            payload = {
                "ticker_data": {
                    "SPY": inference_subset["SPY"].tolist(),
                    "QQQ": inference_subset["QQQ"].tolist(),
                    "GLD": inference_subset["GLD"].tolist(),
                    "0005.HK": inference_subset["0005.HK"].tolist(),
                    "EPHE": inference_subset["EPHE"].tolist()
                }
            }
            st.sidebar.success(f"Data ingested successfully. Formatted {len(inference_subset)} historical steps.")
        except Exception as e: 
            st.error(f"Market Feed Data Collection Failure: {str(e)}")
            st.stop()
    
    with st.spinner("Transmitting data to production FastAPI inference cluster."): 
        try: 
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200: 
                res = response.json()
                results = res["inference_results"]
                portfolio = res["portfolio_allocation"]

                st.subheader("Underlying Input Market Chunks")
                
                metric_cols = st.columns(len(TICKERS))
                
                for idx, ticker in enumerate(TICKERS):
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        final_price = ticker_obj.info.get('regularMarketPrice')
                        prev_close = ticker_obj.info.get('regularMarketPreviousClose')
                        
                        if final_price is None or prev_close is None:
                            final_price = float(inference_subset[ticker].iloc[-1])
                            prev_close = float(inference_subset[ticker].iloc[-2])
                    except Exception:
                        final_price = float(inference_subset[ticker].iloc[-1])
                        prev_close = float(inference_subset[ticker].iloc[-2])
                    
                    nominal_delta = final_price - prev_close
                    pct_delta = (nominal_delta / prev_close) * 100
                    
                    with metric_cols[idx]:
                        st.metric(
                            label=ticker,
                            value=f"{final_price:.2f}",
                            delta=f"{nominal_delta:+.2f} ({pct_delta:+.2f}%)"
                        )
                
                st.markdown("---")
                
                chart_df = inference_subset.div(inference_subset.iloc[0]).mul(100)
                st.line_chart(chart_df, use_container_width=True)
                st.divider()

                st.subheader("System Inference Output")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Inferred Macro Regime", 
                        value=f"State {results['current_regime']}"
                    )
                with col2:
                    st.metric(
                        label="Anomaly Risk Factor Score", 
                        value=f"{results['anomaly_probability'] * 100:.2f}%"
                    )
                with col3:
                    action = portfolio["strategy_action"]
                    st.metric(label="Target Routing Order", value=action)
                
                st.markdown("### Regime Operational Profile")
                if results["current_regime"] in [1, 3]:
                    st.error(f"{results['regime_description']}")
                elif results["current_regime"] == 2:
                    st.warning(f"{results['regime_description']}")
                else:
                    st.success(f"{results['regime_description']}")
                    
                st.divider()
                
                st.subheader("Asset Optimization Target Weights")
                alloc_dict = portfolio["weights"]
                alloc_df = pd.DataFrame({
                    "Asset Ticker": list(alloc_dict.keys()),
                    "Capital Weight (%)": [float(w) * 100 for w in alloc_dict.values()]
                })

                graph_col, table_col = st.columns([2, 1])
                
                with graph_col:
                    st.bar_chart(
                        data=alloc_df, 
                        x="Asset Ticker", 
                        y="Capital Weight (%)", 
                        use_container_width=True
                    )
                with table_col:
                    st.table(alloc_df.set_index("Asset Ticker"))
                    
            else:
                st.error(f"Inference Failure (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Backend Gateway Communication Breakdown: {str(e)}")

else:
    st.info("System Idle. Click 'Fetch Live Market Feed & Run Inference' above to stream real-time vector inputs.")