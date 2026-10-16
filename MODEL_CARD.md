# 模型說明

本公開版本附上先前訓練的兩個五子棋模型，供遊戲 GUI 推論使用。模型架構定義於 policy_value_net.py，推論結合 PUCT MCTS。

## 模型清單

| 檔案 | 棋盤 | Residual blocks | 大小 |
| --- | --- | ---: | ---: |
| `checkpoints/iter_030.pt` | 9×9 | metadata 未記錄，程式預設 0 | 531,381 bytes |
| `checkpoints/curriculum_11x11_deep4_continue_iter_007.pt` | 11×11 | 4 | 5,410,025 bytes |

上述棋盤尺寸及 11×11 模型的 block 數來自 checkpoint ZIP 內的靜態 metadata 檢查，未載入模型執行。9×9 模型的 block 數依載入程式的預設值說明。

## 輸入與輸出

- 輸入：由棋盤局面編碼的四個 feature planes，空間尺寸需與模型棋盤一致。
- 輸出：棋盤落點的 policy 與目前行棋方的 value 估計。
- 選步：MCTS 結合 policy prior 與 value 估計進行搜尋。
- 棋盤使用自由五子棋規則；不處理連珠禁手。

## 來源與限制

模型由本專案先前的自我對弈訓練流程產生。此版本提供自我對弈與戰術課程訓練程式，但不包含舊評估 CSV 或 PDF 報告。

未附上可完整重現這兩個 checkpoint 的隨機種子、訓練環境及逐步設定紀錄，因此不宣稱完整可重現。模型名稱僅供識別，不能單靠檔名推斷總訓練步數或棋力。

checkpoint 保存模型權重與架構資訊，不包含完整 optimizer 狀態。模型尚未在本次整理中進行推論驗證，也沒有經本次驗證的 Elo 或標準 benchmark 分數。

請只載入可信來源的 PyTorch checkpoint；不要將未知 .pt 檔放入 GUI 會掃描的 checkpoints 目錄。

## 完整性

```text
iter_030.pt
SHA-256: 1C5D4FE72AEC477A6692768F6239E4E588E3FED19C0BCE67C31A0B2136E33645

curriculum_11x11_deep4_continue_iter_007.pt
SHA-256: D218E3387C583BC7513DDEC414FC16A762BB7568BFC6F994A5A481D703ACC751
```

## 授權

隨附模型採用根目錄 LICENSE 所列的 MIT License。
