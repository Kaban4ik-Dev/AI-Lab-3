from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Online Retail.xlsx"
OUT_DIR = BASE_DIR / "outputs_b"
OUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42

# --- очистка транзакций ---
MIN_QUANTITY = 1
MIN_UNIT_PRICE = 0

# --- обрезка выбросов перед log1p (winsorize) ---
# 0.99 = клипы на 99-м перцентиле. Значение None отключает клиппинг.
WINSORIZE_QUANTILE = 0.99

# --- отдельная категория «VIP / опт» ---
# клиенты с Monetary или Frequency выше этого квантиля
# исключаются из кластеризации и описываются отдельно
WHOLESALE_QUANTILE = 0.995

# --- k-means ---
KMEANS_KS = list(range(2, 11))       # 2..10
KMEANS_N_INIT = 5
KMEANS_RANDOM_STATE = RANDOM_STATE

# --- Agglomerative ---
AGG_KS = [3, 4, 5, 6, 7]
AGG_LINKAGES = ["ward", "average", "complete"]

# --- DBSCAN ---
DBSCAN_MIN_SAMPLES = [5, 10, 20]
DBSCAN_EPS_QUANTILES = [0.90, 0.93, 0.95, 0.97, 0.99]

# --- GMM ---
GMM_KS = [2, 3, 4, 5, 6]
GMM_COV_TYPES = ["full", "diag"]

# --- PCA ---
PCA_VARIANCE_TARGETS = [0.90, 0.95]

# --- t-SNE ---
# Для отладки можно сократить до [30] и ["pca"].
# Для финального отчёта нужно ≥2 perplexity и ≥2 инициализаций.
TSNE_PERPLEXITIES = [15, 30, 50]
TSNE_INITS = ["pca", "random"]

# --- Устойчивость ---
STABILITY_N_SUBSAMPLES = 5
STABILITY_FRACTION = 0.80
STABILITY_SEED = RANDOM_STATE