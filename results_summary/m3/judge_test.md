| model | n | failed | MAE | exact | Spearman teacher | Spearman utility | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---|
| elicitor-judge-1.7b-sft | 256 | 0 | 0.391 | 0.688 | 0.905 | 0.832 | [0.790, 0.868] |
| teacher | 256 | 0 | – | – | – | 0.871 | [0.827, 0.906] |
| elicitor-qwen3-4b-zs | 256 | 0 | 1.574 | 0.121 | 0.813 | 0.758 | [0.685, 0.819] |
| elicitor-qwen3-1.7b-zs | 256 | 0 | 2.559 | 0.055 | 0.610 | 0.528 | [0.442, 0.607] |
| elicitor-qwen3-0.6b-zs | 256 | 0 | 5.094 | 0.008 | 0.262 | 0.273 | [0.177, 0.354] |

D5 bar: the SFT judge's CI lower bound ≥ 0.8.
