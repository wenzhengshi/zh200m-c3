# SFT 训练数据

这份目录放的是**线上模型实际用来做 SFT 的原始数据**，以及几份评测集。

## 线上模型是怎么来的

```
zh200m-c3-196m-fp32.pt（预训练底座，1.49B token）
        ↓  全参数 SFT，2 epoch，lr 5e-5，本机 i5-4460 CPU，40.5 分钟
zh200m-c3-sft.pt（线上对话模型）
```

训练集 **3,652 条** = `conversation_885.json` 复制两遍（1,770）+ `arithmetic_1882.json`（1,882）。

复制两遍是当时的有意选择：让对话能力和算术能力在训练中权重接近。

## 文件说明

| 文件 | 条数 | 内容 |
|---|---|---|
| `conversation_885.json` | 885 | 中文对话种子，覆盖身份、能力边界、简单问答、拒绝 |
| `arithmetic_1882.json` | 1,882 | 20 以内加减、九九乘法表、表内除法，含表外拒答 |
| `conversation_725.json` | 725 | 更早一版对话种子（885 的前身） |
| `pilot_multiphrasing_195.json` | 195 | 同一问题多问法的实验集，用来验证泛化 |
| `handmade_100.jsonl` | 100 | 手写高质量指令数据，每条覆盖不同场景（未参与线上模型训练） |
| `eval_29.json` | 29 | 留出评测集（对话+简单算术） |
| `pilot_eval_40.json` | 40 | 多问法评测集，A 组同义改写 / B 组全新问法 |

## 格式

对话数据（`.json`）是列表，每条是问答对：

```json
[{"u": "你是谁？", "a": "我是 Zh200M-C3，一个 196M 参数的中文小模型。"}]
```

也可以是多轮（`sft_train.py` 支持）：

```json
[{"turns": [{"u": "...", "a": "..."}, {"u": "...", "a": "..."}]}]
```

手写数据（`.jsonl`）是 Alpaca 风格，每条一行：

```json
{"instruction": "用户的指令", "input": "可选上下文", "output": "回答"}
```

转训练格式的方式：

```python
u = instruction + ("\n" + input if input else "")
a = output
```

## 复现训练

```bash
python3 sft_train.py \
  --data sft/online_3652.json \
  --base zh200m-c3-196m-fp32.pt \
  --out zh200m-c3-sft.pt \
  --epochs 2 --batch 4 --lr 0.00005 --threads 4
```

`online_3652.json` 需要自己合成：`conversation_885.json` 内容重复两次，再接上 `arithmetic_1882.json`。

## 已知局限

- 算术只覆盖表内组合，表外会瞎答（实测表内约 30%、表外拒答约 20%）
- 上下文超过 8 轮或 2048 token 会丢上文
- 换个问法容易答错，说明更多是记忆而非理解

## 数据来源与许可

- `conversation_*.json`、`arithmetic_1882.json`、`handmade_100.jsonl`：本项目自行编写，随仓库以 MIT 许可发布
- 评测集同理，可自由使用

外部语料（CLUECorpusSmall、维基、文库古籍）只用于预训练，不在本目录，见仓库根的 `DATA_NOTICE.md`。
