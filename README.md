# AIoT HW2：多元線性回歸 × 特徵選取 × 95% 預測區間

用兩個 Kaggle 資料集做多元線性回歸，依 CRISP-DM 六步驟完成。網頁會做特徵選取和模型評估，並畫出 95% 信賴區間（CI）與預測區間（PI）的預測圖。

**線上網頁：** https://prometheans152.github.io/AIoT-HW2-MultipleLR/ （GitHub repo 建立並開啟 Pages 後生效）
**完整報告：** [REPORT.md](REPORT.md)

所有運算都在瀏覽器裡跑（HTML + CSS + JavaScript），不需要伺服器或 API 金鑰。也可以直接打開 `index.html` 使用。

## 作業要求對照

| 要求 | 做法 | 位置 |
|---|---|---|
| Kaggle 資料集，特徵 > 13 個 | Car Price 24 個特徵、Student Performance 32 個特徵 | 網頁 01、[REPORT §2](REPORT.md#2-data-understanding資料理解) |
| 線性回歸 | OLS 多元線性回歸（QR 分解求解） | 網頁 03–04 |
| 特徵選取 | Pearson、SelectKBest、RFE、Forward SFS、Lasso，各自用訓練集 5-fold CV 選 k；另附 LassoCV | 網頁 03 |
| 模型評估 | 80/20 切分，報告 Test RMSE、MAE、R²、CV RMSE、VIF、殘差 | 網頁 03–04、[REPORT §5](REPORT.md#5-evaluation評估) |
| 預測圖＋信賴/預測區間 | 測試集預測值與 95% CI、PI 帶狀圖，並算 PI 覆蓋率 | 網頁 05 |
| CRISP-DM 六步驟 | 網頁 07 依目前題目即時產生，報告逐步寫完整 | 網頁 07、[REPORT.md](REPORT.md) |
| GPT 輔助內容 | 見下方〈GPT 輔助內容〉 | 本頁 |
| NotebookLM 摘要 | 見下方〈NotebookLM 摘要〉 | 本頁 |
| 加分 +1：下拉選單兩個題目 | Problem 1 Car Price／Problem 2 Student Performance | 網頁 01 |
| 加分 +2：輸入任意資料集 | 上傳 CSV 或貼公開 CSV 網址（例：50_Startups.csv），全部分析自動重跑 | 網頁 01 |
| GitHub＋網頁 | GitHub Pages，推送到 `main` 時先跑測試再部署 | `.github/workflows/pages.yml` |

## 兩個題目

| | Problem 1：Car Price | Problem 2：Student Performance |
|---|---|---|
| Kaggle | [hellbuoy/car-price-prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) | [uciml/student-alcohol-consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)（`student-mat.csv`） |
| 問題 | 依車輛規格預測美國市場車價 `price` | 依學生背景與前兩次成績預測數學期末成績 `G3`（0–20 分） |
| 資料 | 205 筆、26 欄、無缺值 | 395 筆、33 欄、`;` 分隔、無缺值 |
| 特徵 | 24 個（不含 `car_ID`）；`CarName` 有 147 種值，跳過，實際用 23 個，編碼後 38 欄 | 32 個，編碼後 41 欄 |
| 檔案來源 | Kaggle 需登入，改從 [GitHub 公開副本](https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv) 下載；另一個副本內容完全相同，筆數與欄位和 Kaggle 說明一致 | 從 [UCI 官方 zip](https://archive.ics.uci.edu/dataset/320/student+performance) 下載；Kaggle 版本是同一份 UCI 資料 |

## 結果摘要

每種方法只用訓練集的 5-fold CV RMSE 決定 k，測試集（20%）只用來最後評估。

| 題目 | 方法 | CV 選的 k | Test RMSE | Test R² | 95% PI 覆蓋率 |
|---|---|---:|---:|---:|---:|
| Car Price | Pearson / SelectKBest | 18 | 3,479.64 | 0.8466 | 37/41 = 90.2% |
| | RFE | 5 | 3,594.96 | 0.8363 | 38/41 = 92.7% |
| | Forward SFS | 28 | 4,026.00 | 0.7947 | 35/41 = 85.4% |
| | Lasso 排序 | 12 | 3,624.16 | 0.8336 | 37/41 = 90.2% |
| | LassoCV（原生 L1） | 21 個非零 | 3,435.15 | 0.8505 | — |
| Student（含 G1、G2） | Pearson / SelectKBest | 3 | 2.13 | 0.7782 | 74/79 = 93.7% |
| | RFE | 11 | 2.32 | 0.7378 | 73/79 = 92.4% |
| | Forward SFS | 19 | 2.26 | 0.7499 | 74/79 = 93.7% |
| | Lasso 排序 | 9 | 2.25 | 0.7542 | 74/79 = 93.7% |
| Student（拿掉 G1、G2） | 五種方法 | 3–17 | 4.31–4.32 | 0.088–0.094 | 74–75/79 |

重點：

1. **Car Price**：只用 5 個特徵的 RFE 模型，測試 R² 0.836，和 18 個特徵的 Pearson 模型（0.847）差不多。Forward SFS 的 CV 分數最好，測試卻最差，表示它用 28 個特徵有點過度配適。
2. **Student**：`G2`、`G1`（前兩次成績）幾乎決定了期末成績。拿掉後 R² 從約 0.75 掉到約 0.09，也就是只用背景資料很難預測。
3. **預測區間**：Student 的 5 個 PI 落點外有 4 個是 `G3 = 0`（應是沒參加期末考），這種情況線性模型預測不到。

## 使用方式

1. 在「資料與設定」用下拉選單切換 Problem 1 / Problem 2，或上傳 CSV、貼 CSV 網址（按「試試範例」會載入 50_Startups.csv）。
2. 在「特徵選取」選方法與 k，點折線上的點可以看該模型的特徵與分數。
3. 在「預測區間」看測試集的 CI／PI 帶狀圖與覆蓋率。
4. 在「即時預測」改輸入值，馬上算出預測值和 95% CI、PI。
5. 在「CRISP-DM」看依目前題目自動產生的六步驟摘要；也可以匯出預測 CSV（含區間）與 JSON 報告。

自訂 CSV：第一列是欄位名稱；逗號、分號或 Tab 分隔都可以；至少 20 筆有效資料。類別欄位 2–10 類會做 One-hot，出現少於 5 次的類別合併成「其他」，名稱像 `xxx_id` 且每列不同的欄位會跳過。編碼後最多 60 個特徵。貼網址時，對方網站必須允許跨網域讀取（CORS），例如 `raw.githubusercontent.com`。

## 驗證

```bash
npm ci
```

```bash
npm run test:all
```

```bash
npm run verify
```

- `npm run test:all`：266 個計算斷言（含 t 分位數、區間公式、兩個資料集的結果）和 46 個網頁互動斷言（下拉切換、URL 載入、區間圖、CRISP-DM）。
- `npm run verify`：用 pandas、scikit-learn、SciPy 獨立重算一次。比對了編碼、`train_test_split(random_state=42)`、係數、測試指標、CV RMSE、Pearson／f_regression 排序、Forward 路徑、CI／PI 上下界。結果在 [verify/crosscheck_result.json](verify/crosscheck_result.json)，三組資料全部一致（係數誤差 < 1e-11）。

## 專案結構

```text
index.html / styles.css / app.js   網頁與互動
analysis.js                        回歸、特徵選取、CV、t 分布、CI／PI（無外部套件）
data.js                            內建兩個題目的資料（讓 file:// 也能直接開）
data/                              原始 CSV：CarPrice_Assignment.csv、student-mat.csv、50_Startups.csv
tests/                             Node 測試
verify/                            Python 獨立驗證
REPORT.md                          CRISP-DM 完整報告
```

## GPT 輔助內容

> 標示：以下是生成式 AI 協助的部分。

- 本專案的程式碼、資料檢查和報告初稿，是在 AI 程式助理（Claude Code）協助下完成的。
- AI 協助的部分：
  - 找 Kaggle 資料的公開副本，並核對筆數和欄位。
  - 寫 CI／PI 與特徵選取的程式。
  - 寫測試和 Python 驗證腳本。
  - 整理 CRISP-DM 報告文字。
- 本人做的部分：
  - 選題目。
  - 決定換掉不適合的資料集。原本考慮的 Flood Prediction 資料，目標剛好是特徵總和的 0.005 倍，線性回歸會完全命中，沒有分析意義。
  - 閱讀並確認報告內容。
- 報告中的數字都由程式實際算出，並經過 scikit-learn 獨立驗證；沒有人工填寫的結果。
- **ChatGPT 對話紀錄 PDF：** `docs/GPT對話紀錄.pdf`（待補：請匯出與 ChatGPT 討論本作業的對話後放入）

## NotebookLM 摘要

> 標示：以下是 NotebookLM 產生的摘要。

**待補。** 這段必須由 NotebookLM 實際產生，不可以自己編寫。

建議步驟：

1. 加入以下來源：
   - 兩個 Kaggle 資料集頁面。
   - 2–3 個 Kaggle 上公開的 Car Price 或 Student Performance 回歸 notebook。
   - scikit-learn 的 Feature selection 文件。
   - Cortez & Silva (2008)《Using Data Mining to Predict Secondary School Student Performance》。
2. 對 NotebookLM 提問：「請用繁體中文、至少 100 字，摘要這些網路上的解法：他們用了哪些特徵選取方法、模型評估指標，以及常見的結論。」
3. 把回答貼在這裡，並保留上方的「NotebookLM 摘要」標示。
