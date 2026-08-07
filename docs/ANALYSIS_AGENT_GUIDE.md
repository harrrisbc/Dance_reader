# 分析 Agent 使用指南（Analysis agent）

專門用嚟 **讀／比較** Dance Analyser 產出嘅 `.dance.json`（舞蹈數據），例如兩段舞、兩種風格、兩個舞者。  
唔負責改 App 功能（嗰啲交 design／programming agent）。

| | |
|--|--|
| Cursor rule | `.cursor/rules/analysis-agent.mdc`（開 chat 時啟用呢條 rule） |
| 公式速查 | [CHEATSHEET_METRICS_MOTIFS_YUE.md](./CHEATSHEET_METRICS_MOTIFS_YUE.md) |
| 英文速查 | [CHEATSHEET_METRICS_MOTIFS.md](./CHEATSHEET_METRICS_MOTIFS.md) |
| 寫作輸出（可選） | `docs/analysis/COMPARE_*.md` |

---

## 點開呢個 Agent

1. 開一個 **新 Cursor chat**（建議同 design／programming 分開）
2. 啟用 rule：**Dance Analyser — analysis agent**（`analysis-agent.mdc`）
3. 貼上 JSON 路徑（可以兩個），講你想比較咩

**開場示例：**

```text
用 analysis agent。比較呢兩份：
A: C:\Users\USER\Downloads\07 走路+武拜廟.dance.json   （廟／拜）
B: C:\Users\USER\Downloads\hi.dance.json                 （行路準備）
我想知空間用法同重複動作有咩分別。
```

或者：

```text
分析呢份 JSON 嘅 energy / reach / motif，摘要畀我；之後會再加第二份做對照。
```

---

## 佢會幫你做咩／唔會做咩

| 會 | 唔會 |
|----|------|
| 讀 `.dance.json` 入面 metrics、actors、motifTypes | 重跑 YOLO／改 threshold |
| 兩份（或多份）對照：力度、空間、走位、motif | 當 `pos_x/y` 係舞台米數 |
| 用秒數錨點，方便你喺 Qt 撳去睇 | 用 UI 嘅 % match 當「科學分數」 |
| 用粵語解釋（除非你要求英文） | 亂估片入面「感覺」而 JSON 冇證據 |
| 你要求就寫報告入 `docs/analysis/` | 大幅改 product code |

---

## `.dance.json` 入面有咩（分析時會用）

頂層大概係 `DanceProject` → `analysis`（`AnalysisResult`）：

| 欄位 | 用途 |
|------|------|
| `duration`, `fps`, `frame_width/height` | 片長、取樣、畫面比例 |
| `actors[].metrics[]` | 每幀曲線：`energy`, `speed`, `reach`, `level`, `travel`, `directness`, `pos_x`, `pos_y` |
| `actors[].poses[]` | 骨架（通常好大；比較曲線時唔使逐幀讀晒） |
| `motifTypes[]` | 自動重複類型 + `occurrences`（起止秒） |
| `motifDiag` | 偵測摘要字串（有就睇） |
| `manualMarks` | 人手標記（如有） |

曲線單位同計法見速查表。記住：

- **Energy** = 相對呢個舞者自己高峰（唔好直接跨片話邊個「勁過邊個」，除非先講清楚）
- **Level** 而家仍然係「相對本片腳長 p90」（跨風格要小心）
- **Floor / L–R / F–B** = 畫面座標
- **自動 Motif** 偏「寧願少併」；冇 type ≠ 冇重複，可以用 Qt 嘅 Find Similar 再查

---

## 建議比較框架（兩種舞／兩份 JSON）

Agent 預設會跟呢個順序想；你都可以指定只做其中幾項。

### 1. 先對齊條件

- 片長差幾多？fps／解像度一唔一樣？
- 幾個 dancer？揀邊個 `actor`？
- 鏡頭固唔固定？（影響 travel／floor）

### 2. 力度同節奏（Effort）

- Energy／Speed：median、p95、最高峰出現喺邊幾秒
- 「密唔密」：高峰多定疏；有冇大段休息（低 energy 谷）

### 3. 身體形狀同高低（Shape）

- Reach：偏合定開；有冇突然打開
- Level：企／蹲比例（跨片只做相對描述）

### 4. 空間同走位（Space）

- Travel：郁位密唔密
- Directness：直走多定兜圈（企定時會空白）
- `pos_x`／`pos_y` 散佈：偏左／偏前定用晒畫面

### 5. 結構（Motif／phrase）

- 有幾個 motif type？每個出現幾次、喺邊段秒數？
- `meanIntraDist`（如有）：type 內部有幾「散」
- A 有重複結構、B 冇（或相反）→ 講清楚，唔好硬凹

### 6. 結論要可驗證

每個重要對比最好帶 **秒數**，方便你喺 desktop 播返確認。

---

## 報告檔（可選）

你講「寫低」或「存報告」時，Agent 應存：

`docs/analysis/COMPARE_<簡稱>_<YYYYMMDD>.md`

建議結構：

1. 一句總結  
2. 檔案路徑同標籤（風格／作品／舞者）  
3. 各自 snapshot  
4. 對照表（metrics + motif）  
5. 限制／唔可以結論嘅嘢  

---

## 同其他 Agent 點分工

| Role | 做咩 |
|------|------|
| **Analysis**（呢個） | 讀 JSON、比較、解釋、寫分析筆記 |
| **Design** | 產品／指標定義／handoff |
| **Programming** | 改 code、修 bug、跟 handoff 實作 |

如果你分析完發現「計法錯」或「想多一個指標」→ 開 design agent 寫工單，唔好叫 analysis agent 直接改 pipeline。

---

## 小貼士

- 一次比較 **兩份** 最清楚；第三份可以之後再加。  
- 檔好大（poses）時，叫 agent **優先讀 metrics + motifTypes**，唔好無謂 dump 晒 poses。  
- 比較「中國舞 vs 街舞」之類：先用數據對照，再用人話總結風格差異，並標明邊啲係數據、邊啲係解讀。
