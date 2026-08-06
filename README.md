# Dance Analyser

本機舞蹈分析工具（**Python 桌面版為主**）：YOLO-Pose + Supervision、能量／速度／空間指標、重複動作類型。

介面語言：繁體中文。不需要瀏覽器。

## 啟動（建議）

```powershell
cd C:\dev\Dance_reader\Dance_reader\backend
.\.venv\Scripts\activate
pip install -r requirements.txt

cd ..
python -m desktop
```

若 venv 唔喺 PATH：

```powershell
.\backend\.venv\Scripts\python.exe -m desktop
```

（請喺 repo 根目錄執行。）

## 用法

1. **開啟影片** — 本機檔案  
2. **播放／拖曳** — 下方滑桿；可點時間軸圖跳轉；**放大／縮小／重設**（或滾輪）  
3. **最多舞者** — 1–4（預設 2）；丟失後會回收同一舞者編號  
4. **分析（Supervision）** — 預設 `yolo11s-pose`、每幀；能量／速度／伸展／高低／走位／直進、重複類型  
5. **儲存專案** — 存成 `.dance.json`（骨架／曲線／類型一齊）；下次 **開啟專案** 唔使再跑分析；舊專案會由 pose 補算空間曲線  
6. 選舞者、勾選曲線睇圖表；點類型／出現次數跳轉；**上一處／下一處**  
7. 跳去某段動作 → **搵相似** → 列出所有相似出現  

## 時間軸曲線

| 曲線 | 含義 |
|------|------|
| 能量 | 整體關節動能 |
| 速度 | 肢體平均速度 |
| 伸展 | kinesphere／伸展幅度（相對肩寬） |
| 高低 | 髖部高度（高姿 → 高值） |
| 走位 | 短窗內 COM 路徑長度 |
| 直進 | 淨位移 ÷ 路徑（1 = 直線） |

預設顯示：能量、速度、伸展。

## 可選：舊 Web API

`backend` 仍有 FastAPI（除錯用）。日常請用 `python -m desktop`。  
舊 Vite 前端屬 legacy，不必再開。

## YouTube

尚未接入桌面版；請先下載成檔再開啟。之後可用 yt-dlp。
