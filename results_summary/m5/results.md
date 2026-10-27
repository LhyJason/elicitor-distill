### Non-inferiority (Q12): `sel=eig_mc,upd=llm,phr=tpl`, `sel=eig,upd=llm,phr=tpl` at K = 5

| model | metric | model − teacher | 95 % CI | one-sided lower | margin | verdict |
|---|---|---:|---|---:|---:|---|
| elicitor-agent-1.7b-sft (sel=eig,upd=llm,phr=tpl) | mass_on_truth | -0.0036 | [-0.0098, +0.0023] | -0.0087 | -0.02 | non-inferior |
| elicitor-agent-1.7b-sft (sel=eig_mc,upd=llm,phr=tpl) | mass_on_truth | +0.0037 | [+0.0002, +0.0072] | +0.0008 | -0.02 | non-inferior |
| elicitor-upd-1.7b-sft (sel=eig,upd=llm,phr=tpl) | mass_on_truth | +0.0095 | [+0.0047, +0.0148] | +0.0054 | -0.02 | non-inferior |
| elicitor-upd-1.7b-sft (sel=eig_mc,upd=llm,phr=tpl) | mass_on_truth | +0.0092 | [+0.0020, +0.0172] | +0.0029 | -0.02 | non-inferior |
| elicitor-agent-1.7b-sft (sel=eig,upd=llm,phr=tpl) | map_accuracy | -0.0019 | [-0.0060, +0.0019] | -0.0051 | -0.03 | non-inferior |
| elicitor-agent-1.7b-sft (sel=eig_mc,upd=llm,phr=tpl) | map_accuracy | +0.0028 | [-0.0019, +0.0069] | -0.0009 | -0.03 | non-inferior |
| elicitor-upd-1.7b-sft (sel=eig,upd=llm,phr=tpl) | map_accuracy | -0.0028 | [-0.0079, +0.0014] | -0.0069 | -0.03 | non-inferior |
| elicitor-upd-1.7b-sft (sel=eig_mc,upd=llm,phr=tpl) | map_accuracy | -0.0083 | [-0.0218, +0.0060] | -0.0199 | -0.03 | non-inferior |

### Every comparison (Holm per metric over model × arm × budget)

| metric | model | arm | K | model | teacher | diff | 95 % CI | p (Holm) |
|---|---|---|---:|---:|---:|---:|---|---:|
| mass_on_truth | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.5098 | 0.5135 | -0.0036 | [-0.0098, +0.0023] | 0.445 |
| mass_on_truth | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.5443 | 0.5406 | +0.0037 | [+0.0002, +0.0072] | 0.0397 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.5230 | 0.5135 | +0.0095 | [+0.0047, +0.0148] | 0.00214 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.5498 | 0.5406 | +0.0092 | [+0.0020, +0.0172] | 0.307 |
| map_accuracy | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.6537 | 0.6556 | -0.0019 | [-0.0060, +0.0019] | 0.688 |
| map_accuracy | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.6704 | 0.6676 | +0.0028 | [-0.0019, +0.0069] | 0.618 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.6528 | 0.6556 | -0.0028 | [-0.0079, +0.0014] | 0.688 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.6593 | 0.6676 | -0.0083 | [-0.0218, +0.0060] | 0.618 |
| log_loss | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2002 | 1.1951 | +0.0051 | [-0.0110, +0.0210] | 0.993 |
| log_loss | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 1.1422 | 1.1559 | -0.0137 | [-0.0233, -0.0044] | 0.00505 |
| log_loss | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.1648 | 1.1951 | -0.0303 | [-0.0443, -0.0177] | 0.000281 |
| log_loss | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 1.1485 | 1.1559 | -0.0073 | [-0.0276, +0.0113] | 0.993 |
| regret | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.1100 | 0.1116 | -0.0015 | [-0.0046, +0.0012] | 1 |
| regret | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.1060 | 0.1096 | -0.0035 | [-0.0113, +0.0042] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.1102 | 0.1116 | -0.0014 | [-0.0060, +0.0025] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.1210 | 0.1096 | +0.0114 | [-0.0106, +0.0327] | 0.985 |
| hit@1 | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.2167 | 0.2208 | -0.0042 | [-0.0250, +0.0125] | 1 |
| hit@1 | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.2250 | 0.2167 | +0.0083 | [-0.0250, +0.0417] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.2250 | 0.2208 | +0.0042 | [-0.0125, +0.0250] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.1917 | 0.2167 | -0.0250 | [-0.1167, +0.0583] | 1 |
| violations | elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2250 | 1.2375 | -0.0125 | [-0.0417, +0.0167] | 1 |
| violations | elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.3292 | 0.3375 | -0.0083 | [-0.0500, +0.0333] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2208 | 1.2375 | -0.0167 | [-0.0667, +0.0250] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 0.4583 | 0.3375 | +0.1208 | [+0.0000, +0.2417] | 0.293 |

### Episodes that asked the teacher's questions in the teacher's order

| model | arm | K | episodes | same |
|---|---|---:|---:|---:|
| elicitor-agent-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-agent-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 240 | 0.200 |
| elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=eig_mc,upd=llm,phr=tpl | 5 | 240 | 0.000 |
