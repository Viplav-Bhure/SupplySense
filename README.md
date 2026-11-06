# SupplySense

SupplySense helps a shop decide what to order and when. It forecasts demand for the next 7 days with an LSTM, rates each product's stockout risk with a Bayesian network, flags unusual sales with a variational autoencoder, and turns those results into reorder points, safety stock and recommended order quantities, all in an interactive Streamlit dashboard.

## Pages

| Page | What it does |
| --- | --- |
| Overview | Key figures, risk breakdown, products needing attention, position vs reorder point |
| Demand forecast | 7-day forecast per product, with a CSV download |
| Inventory | Filterable table of reorder points, safety stock, risk, stockout chance, recommended orders and the reason for each status |
| Anomalies | Unusual sales found by the VAE (or a z-score fallback), each with a "why flagged" explanation |
| What-if | Demand change and supplier delay scenarios compared with the baseline |
| Data & Models | Choose a data source, **upload your own data**, edit inventory inputs, train the models |
| Glossary | Plain-language definitions of every term used in the app |

The **Settings** button on every page controls the target service level, the review period and the forecast engine.

## Tech stack

Python, PyTorch (LSTM and VAE), pgmpy (Bayesian network), pandas and NumPy, Streamlit and Plotly.

## Quick start

Requires Python 3.10 or newer (developed and tested on Python 3.12 with Streamlit 1.66).

```bash
git clone https://github.com/Viplav-Bhure/SupplySense.git
cd SupplySense
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

1. **Get the data (optional).** Download `train.csv` from the Kaggle competition [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (you may need to join it first) and put it in `data/`. Without it the app uses synthetic sample data.
2. **Train the models (once).** This can take several minutes on a CPU. Skip it if `models/saved/` already contains `lstm.pt` and `vae.pt`.
   ```bash
   python -m models.lstm_forecaster
   python -m models.vae_anomaly
   ```
   Without trained models the app still runs, using a moving-average forecast and a z-score anomaly detector.
3. **Run the app.**
   ```bash
   streamlit run app.py
   ```

Optional checks: `python -m inventory.calculator` prints a worked example (reorder point 153) and `python -m models.bayesian_risk` prints a sample risk assessment.

## Project structure

```
SupplySense/
├── app.py                    Streamlit app: pages, navigation, caching
├── requirements.txt
├── assets/logo.svg
├── .streamlit/config.toml    dark theme
├── data/                     put train.csv here; inventory.csv is created automatically
├── inventory/
│   └── calculator.py         reorder point, safety stock, order quantity, status
├── models/
│   ├── lstm_forecaster.py    7-day demand forecast (PyTorch)
│   ├── bayesian_risk.py      stockout risk network (pgmpy)
│   ├── vae_anomaly.py        anomaly detection (PyTorch)
│   └── saved/                trained model files, created by the training commands
└── utils/
    ├── preprocessing.py      data loading, cleaning, windowing
    ├── uploads.py            upload parsing and validation
    └── ui.py                 theme CSS, KPI cards, chart styling
```

## Data

**Built-in:** the Kaggle *Store Item Demand Forecasting Challenge* (`demand-forecasting-kernels-only`): daily sales for 10 stores and 50 items over five years. Each (store, item) pair is one product; 100 of the 500 pairs are used (`N_PRODUCTS` in `utils/preprocessing.py`, spread evenly over stores and items).

**Your own data:** open *Data & Models* and choose *Upload my own*.

- Sales file (CSV or Excel): one row per product per day or per sale, with a date, a product and a quantity column. You choose the columns after uploading, and common names are detected automatically. At least 90 days of history are needed, and at most 60 products (the highest-volume ones) are kept.
- Inventory file (optional): `product_id, current_stock, on_order, backorders, lead_time_days, supplier_delay_days`. Missing columns get defaults and missing products are simulated. Templates can be downloaded on the page.

The Kaggle dataset has sales only. Stock levels, lead times and supplier delays are **simulated** (fixed random seed) whenever you don't supply them, so treat the inventory recommendations as a demonstration of the pipeline.

## How it works

**Demand forecast (LSTM).** The last 28 days of sales predict the next 7 days. One global model (2 LSTM layers, 64 hidden units) is trained on all products. The data is split chronologically, 80% for training and 20% for validation, and the model is compared with a seasonal-naive baseline that repeats the last week.

**Inventory engine.** Assumes roughly normal daily demand.

- Delivery wait = supplier lead time + current supplier delay
- Safety stock = z × demand standard deviation × √(delivery wait), with z from the target service level
- Reorder point = expected demand during the wait + safety stock
- Inventory position = stock on hand + on order − backorders
- Target level = reorder point + demand over the review period; order quantity = target level − position (minimum 0)
- Status: **Critical** = position < expected demand during the wait; **Reorder** = position < reorder point
- Stockout chance = probability that demand during the wait exceeds the position

**Risk level (Bayesian network).** Points are added up: stock cover (counted 4×: high 0, medium 1, low 2), demand level (0 to 2) and supplier delay (0 to 2). Low stock cover is always High risk; medium cover becomes High when demand and delay add up to 3 or more; high cover becomes Medium only when demand is High and the delay is Long.

**Anomaly detection (VAE).** Windows of 28 days of sales, each scaled by its own mean, are reconstructed by a variational autoencoder. A window with reconstruction error above the 99th percentile of the training errors is flagged, and the day that differs most from the usual level is reported. Deviations under 20% are hidden. Severity is High at 60% or more, Medium at 35% or more, and Low otherwise.

## Training

Models are saved to `models/saved/` and picked up by the app automatically. You can also train from the *Data & Models* page; those models last only for the browser session.

```bash
python -m models.lstm_forecaster
python -m models.vae_anomaly
```

After changing the dataset, retrain both models so they match the data. The LSTM prints its validation MAE, RMSE and MAPE next to the seasonal-naive baseline.

## Limitations

- Stock levels, lead times and supplier delays are simulated when the dataset does not contain them.
- Demand is assumed to be normally distributed, and the LSTM produces a single point forecast rather than a probability distribution.
- The Bayesian network's probability table is set by rules, not learned from labelled stockout history.
- The forecast uses past sales only; prices, promotions and holidays are not included.
- The data has no anomaly labels, so the VAE alerts cannot be scored with precision or recall.

## Author

Built by Viplav Bhure.