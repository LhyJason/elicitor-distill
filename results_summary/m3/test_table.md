### update

| model | seed | n | failed | agreement | mae | spearman | kl | unconstrained_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elicitor-qwen3-0.6b-zs | 0 | 4086 | 0 | 0.480 | 3.768 | 0.500 | 0.554 | – |
| elicitor-qwen3-1.7b-zs | 0 | 4086 | 0 | 0.921 | 1.576 | 0.806 | 0.289 | – |
| elicitor-qwen3-4b-zs | 0 | 4086 | 0 | 0.956 | 1.460 | 0.844 | 0.246 | – |
| elicitor-upd-0.6b-sft | 0 | 4086 | 0 | 0.976 | 0.393 | 0.961 | 0.073 | – |
| elicitor-upd-1.7b-sft | 0 | 4086 | 0 | 0.980 | 0.397 | 0.966 | 0.072 | – |
| elicitor-upd-4b-sft | 0 | 4086 | 0 | 0.987 | 0.297 | 0.974 | 0.056 | – |

### judge

| model | seed | n | failed | mae | exact | within_1 | spearman_teacher | spearman_utility | unconstrained_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elicitor-judge-1.7b-sft | 0 | 256 | 0 | 0.391 | 0.688 | 0.926 | 0.905 | 0.832 | – |
| elicitor-qwen3-0.6b-zs | 0 | 256 | 0 | 5.094 | 0.008 | 0.035 | 0.262 | 0.273 | – |
| elicitor-qwen3-1.7b-zs | 0 | 256 | 0 | 2.559 | 0.055 | 0.234 | 0.610 | 0.528 | – |
| elicitor-qwen3-4b-zs | 0 | 256 | 0 | 1.574 | 0.121 | 0.484 | 0.813 | 0.758 | – |
