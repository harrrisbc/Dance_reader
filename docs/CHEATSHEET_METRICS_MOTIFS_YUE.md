# 速查表 — 數據同 Motif 點計

對應而家 pipeline（`metrics.py` + `motifs.py`）。  
英文版：[CHEATSHEET_METRICS_MOTIFS.md](./CHEATSHEET_METRICS_MOTIFS.md)

所有長度都先將 keypoint 校正做 **square-pixel**（`x' = x · W/H`），先至計。

---

## 0. 共用幾何（成條數都企喺呢度）

| 符號 | 意思 |
|------|------|
| 軀幹中心 `(cx, cy)` | 左右肩 + 左右髖嘅中點（COCO 5,6,11,12） |
| Scale `S` | 軀幹高度（肩中 → 髖中）；有 **floor**（跟住條 track 嘅 median），避免接近 0 時數字爆炸 |
| Visible | confidence ≥ `TORSO_CONF`（0.2）先算睇到；遮擋關節係 **mask 走**，唔係寫成 (0,0) |

配對用嘅 pose feature：關節 XY 相對中心再 ÷ `S`，再加骨骼單位向量（rich feature）。  
Mirror = 左右對調同一套 feature（左右對稱當同一個 gesture）。

---

## 1. 每格曲線（Charts）

喺 `compute_metrics` → `MetricPoint`。  
Energy 同 Speed 會做 **Savitzky–Golay** 平滑（約 83 ms）；**圖表顯示**同 **切 phrase** 用同一條曲線。

| Field | 圖表名 | 點計（人話） | 圖上單位 |
|-------|--------|--------------|----------|
| `energy` | Energy | 肢體關節 `(速度 / S)²` 加總，再 ÷ 呢個舞者自己嘅 **最高峰** | 0–1 相對值 |
| `speed` | Speed | 同樣關節嘅平均 `\|v\| / S` | 軀幹高度 / 秒 |
| `reach` | Reach | 頭／手腕／腳踝離軀幹中心最遠距離 ÷ `S` | × 軀幹（約 1.2 合／2.5 開） |
| `level` | Level | 平均 `\|踝_y − 髖_y\|` ÷ 呢條 track 腳長嘅 **90 百分位**，夾喺 0–1 | 站姿比例（*見註*） |
| `travel` | Travel | 約 **0.8 秒** 窗口內腳跡路徑長 ÷ 平均 `S` ÷ 秒數 | 軀幹高度 / 秒 |
| `directness` | Directness | 窗口內 `淨位移 / 路徑長`；幾乎企定就 **空白（NaN）** | 0 = 彎嚟彎去，1 = 直走 |
| `pos_x` | L–R | 軀幹 COM 嘅畫面 x | 0 = 左，1 = 右 |
| `pos_y` | F–B | 支撐腳（較低嗰隻踝）嘅畫面 y | 0 = 上／遠，1 = 下／近 |

**Travel／Directness 路徑**（跟腳，有 aspect 校正）：

```text
path = Σ hypot(Δx·aspect, Δy)     喺滾動窗口入面加
net  = hypot( 尾 − 頭 )
travel     = (path / S_窗口) / Δt
directness = net / path            如果 path 夠長；唔夠就留空
```

**注意**

- 偵測唔穩／冇 pose 嗰陣，會 **沿用上一個好數值**（避免曲線尖峰）。
- 開舊 `.dance.json` 會用 `enrich_metrics` 由 poses **重計晒**曲線。
- **Level 而家仍然**係「腳長 ÷ 本片 p90」。之後會改做 `(支撐踝_y − 髖_y) / 軀幹高`（工單 `AGENT_HANDOFF_LEVEL_DEFINITION.md`）。
- Floor map 用 `pos_x`／`pos_y` — **畫面座標**，唔係舞台米數。

---

## 2. Motif — 自動類型（「重複動作」）

流程：**切句 → 特徵 → 距離 → 分群 → 保留**。

```text
poses
  → 喺 Energy 谷底切 phrase（真休息）
  → 丟掉太靜嘅 phrase（motion gate）
  → 每句重採樣做 14 格（連 visibility mask）
  → rich pose feature（另存左右鏡像版）
  → 兩兩距離（先 cosine 過濾，再 mask-aware DTW）
  → complete-linkage 分群
  → 只保留次數夠、內部夠緊嘅 type
```

### 2.1 點切 phrase

1. 用同圖表一樣、平滑過嘅 energy（仲未做 peak-normalize）。
2. 搵局部最低點，而且低過附近高峰 × `VALLEY_FRAC`（0.55）— 真係停一下，唔係微震。
3. 長度大約夾喺 **0.65–2.8 秒**；谷底切唔掂就改用約 1.2 秒滑動窗。
4. **Motion gate：** 速度唔夠嘅 phrase 丟掉。

### 2.2 一句點表示

- 重採樣成 `SEQ_FRAMES = 14`（呢步會 **丟掉速度／tempo**）。
- 同時存 mirror 序列（左做／右做當同一 gesture）。
- 不夠長要 pad 嘅格係 **零可見**（唔好當真 pose）。

### 2.3 兩句距離點計

```text
1. Embedding 做 cosine；同一同 mirror 都 < 0.55 → 直接拒絕（超大距離）
2. 否則做 DTW；攞 (A vs B) 同 (A vs mirror B) 較好嗰個
3. 每格成本 = 只計雙方都睇到嘅維度嘅 RMS
   - 共同可見關節太少 → 成本至少係 median（遮擋唔可以「平過」）
4. 決策用原始距離 d（細 = 似）
```

畫面上個 **% match** 淨係畀人睇：`score = clip(1 − d/0.50, 0, 1)` — **唔會**用嚟做閘門。

### 2.4 分群 → motif type

| 閘門 | 預設 | 用途 |
|------|------|------|
| `CLUSTER_LINKAGE` | `complete` | 群內最遠嗰對都要過關（防 A~B~C 一串黏埋） |
| `CLUSTER_MAX_DIST` | **0.24** | 分群切割距離 |
| `INTRA_MAX_DIST` | **0.26** | 一個 type 入面平均兩兩距離上限 |
| `MOTIF_MIN_OCCURRENCES` | **3** | NMS 之後最少幾次；短片冇重複可以回 **0 個 type** |

自動類型原則：**寧願少併，唔好亂併**。  
「似但唔同」唔好塞同一個 type；真重複畀邊界切散咗，可以用 Find Similar 搵返。

---

## 3. Find Similar（你主動搵）

同自動偵測故意唔同行。

1. Query = 你揀嘅 occurrence 時段，或者 playhead 附近一個窗口（**唔強制**跟 valley 切法）。
2. **Subsequence DTW** 將 query 喺成條 track 上面滑（頭尾開放）。
3. 攞局部最低點，做 NMS；顯示 tempo 倍率 `hit時長 / query時長`。
4. 倍率唔喺 **約 0.5–2.0** 就丟（同一形狀、快慢差兩倍以內當得）。
5. 硬貨：`d ≤ 0.16`；再鬆啲可以去到 `d ≤ 0.30`。

一句總結：自動類型偏保守；想搵盡就用 Find Similar。

---

## 4. 一頁對照（想睇咩 → 睇邊條）

| 你想知… | 睇… | 大概意思 |
|---------|-----|----------|
| 身體有幾「用力／活躍」 | Energy | 相對自己高峰嘅動能代理 |
| 四肢郁得幾快 | Speed | 軀幹／秒 |
| 身體開唔開 | Reach | 離中心幾多個軀幹 |
| 企得高定蹲得低 | Level | 相對呢條片「最企直」嘅腳長（*之後會改*） |
| 地上行咗幾多 | Travel | 腳跡／軀幹／秒 |
| 直走定兜圈 | Directness | 淨位移／路徑 |
| 喺畫面邊 | L–R／F–B + floor map | 畫面座標 |
| 「呢句有冇再出現」（目錄） | Motif types | 谷底切句再分群 |
| 「幫我喺成條片搵呢一段」 | Find Similar | Subsequence DTW + tempo |

---

## 5. Code 去邊度睇

| 步驟 | 檔案 |
|------|------|
| Scale、feature、曲線 | `backend/app/pipeline/metrics.py` |
| Phrase、DTW、分群、Find Similar | `backend/app/pipeline/motifs.py` |
| Analyze 串埋 | `backend/app/pipeline/run.py` |
| 開檔重計曲線 | `backend/app/project_io.py` → `enrich_metrics` |
| 調 threshold | `backend/scripts/calibrate_motifs.py` |

更細嘅改動紀錄：`docs/IMPLEMENTATION_NOTES_MATCH_W*.md`、`docs/IMPLEMENTATION_NOTES_MOTIF_OVERMERGE.md`。
