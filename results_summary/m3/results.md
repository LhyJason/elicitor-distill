### Non-inferiority (Q12): `sel=eig,upd=llm,phr=tpl` at K = 5

| model | metric | model − teacher | 95 % CI | one-sided lower | margin | verdict |
|---|---|---:|---|---:|---:|---|
| elicitor-upd-1.7b-sft | mass_on_truth | +0.0095 | [+0.0047, +0.0148] | +0.0054 | -0.02 | non-inferior |
| elicitor-upd-4b-sft | mass_on_truth | +0.0061 | [-0.0018, +0.0137] | -0.0005 | -0.02 | non-inferior |
| elicitor-upd-0.6b-sft | mass_on_truth | -0.0017 | [-0.0074, +0.0038] | -0.0065 | -0.02 | non-inferior |
| elicitor-qwen3-4b-zs | mass_on_truth | -0.1241 | [-0.1350, -0.1127] | -0.1334 | -0.02 | not shown |
| elicitor-qwen3-1.7b-zs | mass_on_truth | -0.1024 | [-0.1132, -0.0916] | -0.1113 | -0.02 | not shown |
| elicitor-qwen3-0.6b-zs | mass_on_truth | -0.2333 | [-0.2469, -0.2190] | -0.2448 | -0.02 | not shown |
| elicitor-upd-1.7b-sft | map_accuracy | -0.0028 | [-0.0079, +0.0014] | -0.0069 | -0.03 | non-inferior |
| elicitor-upd-4b-sft | map_accuracy | -0.0005 | [-0.0046, +0.0032] | -0.0037 | -0.03 | non-inferior |
| elicitor-upd-0.6b-sft | map_accuracy | -0.0060 | [-0.0120, -0.0009] | -0.0111 | -0.03 | non-inferior |
| elicitor-qwen3-4b-zs | map_accuracy | -0.0009 | [-0.0074, +0.0051] | -0.0060 | -0.03 | non-inferior |
| elicitor-qwen3-1.7b-zs | map_accuracy | -0.0227 | [-0.0347, -0.0120] | -0.0324 | -0.03 | not shown |
| elicitor-qwen3-0.6b-zs | map_accuracy | -0.1981 | [-0.2472, -0.1514] | -0.2384 | -0.03 | not shown |

### Every comparison (Holm per metric over model × arm × budget)

| metric | model | arm | K | model | teacher | diff | 95 % CI | p (Holm) |
|---|---|---|---:|---:|---:|---:|---|---:|
| mass_on_truth | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.4096 | 0.4050 | +0.0046 | [+0.0009, +0.0086] | 0.32 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.5230 | 0.5135 | +0.0095 | [+0.0047, +0.0148] | 0.00964 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6784 | 0.6677 | +0.0106 | [+0.0052, +0.0167] | 0.0127 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.4037 | 0.4007 | +0.0029 | [+0.0006, +0.0057] | 0.351 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.5174 | 0.5110 | +0.0063 | [+0.0029, +0.0102] | 0.0127 |
| mass_on_truth | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6871 | 0.6775 | +0.0096 | [+0.0047, +0.0150] | 0.0149 |
| mass_on_truth | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.4125 | 0.4050 | +0.0074 | [+0.0026, +0.0128] | 0.0822 |
| mass_on_truth | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.5196 | 0.5135 | +0.0061 | [-0.0018, +0.0137] | 0.2 |
| mass_on_truth | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6787 | 0.6677 | +0.0110 | [+0.0021, +0.0194] | 0.0822 |
| mass_on_truth | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.4042 | 0.4007 | +0.0034 | [-0.0003, +0.0069] | 0.162 |
| mass_on_truth | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.5181 | 0.5110 | +0.0071 | [+0.0023, +0.0116] | 0.0417 |
| mass_on_truth | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6884 | 0.6775 | +0.0109 | [+0.0026, +0.0189] | 0.0475 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.4053 | 0.4050 | +0.0002 | [-0.0031, +0.0036] | 1 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.5118 | 0.5135 | -0.0017 | [-0.0074, +0.0038] | 1 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6650 | 0.6677 | -0.0028 | [-0.0085, +0.0028] | 1 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.3985 | 0.4007 | -0.0022 | [-0.0053, +0.0006] | 1 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.5095 | 0.5110 | -0.0016 | [-0.0050, +0.0016] | 1 |
| mass_on_truth | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6751 | 0.6775 | -0.0024 | [-0.0078, +0.0028] | 1 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.3232 | 0.4050 | -0.0818 | [-0.0893, -0.0736] | 3.86e-08 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.3894 | 0.5135 | -0.1241 | [-0.1350, -0.1127] | 3.86e-08 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.5000 | 0.6677 | -0.1678 | [-0.1819, -0.1530] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.3348 | 0.4007 | -0.0659 | [-0.0714, -0.0603] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.4041 | 0.5110 | -0.1069 | [-0.1147, -0.0987] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.5072 | 0.6775 | -0.1703 | [-0.1831, -0.1571] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.3525 | 0.4050 | -0.0525 | [-0.0598, -0.0446] | 5.51e-08 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.4111 | 0.5135 | -0.1024 | [-0.1132, -0.0916] | 3.86e-08 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.5121 | 0.6677 | -0.1557 | [-0.1699, -0.1414] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.3420 | 0.4007 | -0.0588 | [-0.0650, -0.0525] | 3.86e-08 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.4131 | 0.5110 | -0.0980 | [-0.1072, -0.0883] | 3.55e-13 |
| mass_on_truth | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.5176 | 0.6775 | -0.1599 | [-0.1727, -0.1467] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.2665 | 0.4050 | -0.1385 | [-0.1488, -0.1271] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.2801 | 0.5135 | -0.2333 | [-0.2469, -0.2190] | 3.86e-08 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.2917 | 0.6677 | -0.3760 | [-0.3925, -0.3590] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.2613 | 0.4007 | -0.1394 | [-0.1459, -0.1327] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.2819 | 0.5110 | -0.2292 | [-0.2396, -0.2184] | 2.56e-13 |
| mass_on_truth | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.3115 | 0.6775 | -0.3660 | [-0.3820, -0.3498] | 2.56e-13 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.4991 | 0.5023 | -0.0032 | [-0.0079, -0.0005] | 1 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.6528 | 0.6556 | -0.0028 | [-0.0079, +0.0014] | 1 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.8940 | 0.8995 | -0.0056 | [-0.0144, +0.0019] | 1 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.4912 | 0.4931 | -0.0019 | [-0.0051, +0.0009] | 1 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.6505 | 0.6542 | -0.0037 | [-0.0088, +0.0005] | 1 |
| map_accuracy | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.8921 | 0.8977 | -0.0056 | [-0.0144, +0.0019] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.5023 | 0.5023 | +0.0000 | [-0.0014, +0.0014] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.6551 | 0.6556 | -0.0005 | [-0.0046, +0.0032] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.8986 | 0.8995 | -0.0009 | [-0.0069, +0.0042] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.4935 | 0.4931 | +0.0005 | [-0.0019, +0.0028] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.6528 | 0.6542 | -0.0014 | [-0.0060, +0.0019] | 1 |
| map_accuracy | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.8968 | 0.8977 | -0.0009 | [-0.0069, +0.0042] | 1 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.4977 | 0.5023 | -0.0046 | [-0.0097, -0.0009] | 1 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.6495 | 0.6556 | -0.0060 | [-0.0120, -0.0009] | 1 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.8917 | 0.8995 | -0.0079 | [-0.0167, -0.0005] | 1 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.4894 | 0.4931 | -0.0037 | [-0.0088, +0.0005] | 1 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.6477 | 0.6542 | -0.0065 | [-0.0130, -0.0009] | 0.352 |
| map_accuracy | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.8903 | 0.8977 | -0.0074 | [-0.0162, +0.0000] | 1 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.5028 | 0.5023 | +0.0005 | [-0.0028, +0.0037] | 1 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.6546 | 0.6556 | -0.0009 | [-0.0074, +0.0051] | 1 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.8755 | 0.8995 | -0.0241 | [-0.0380, -0.0116] | 0.0267 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.4787 | 0.4931 | -0.0144 | [-0.0222, -0.0069] | 0.0185 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.6352 | 0.6542 | -0.0190 | [-0.0292, -0.0102] | 0.0113 |
| map_accuracy | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.8690 | 0.8977 | -0.0287 | [-0.0426, -0.0157] | 0.00528 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.4875 | 0.5023 | -0.0148 | [-0.0250, -0.0065] | 0.0117 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.6329 | 0.6556 | -0.0227 | [-0.0347, -0.0120] | 0.0149 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.8176 | 0.8995 | -0.0819 | [-0.1042, -0.0611] | 5.57e-06 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.4644 | 0.4931 | -0.0287 | [-0.0380, -0.0204] | 9.22e-05 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.6060 | 0.6542 | -0.0481 | [-0.0620, -0.0352] | 1.44e-05 |
| map_accuracy | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.8218 | 0.8977 | -0.0759 | [-0.0977, -0.0556] | 7.17e-06 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.3944 | 0.5023 | -0.1079 | [-0.1505, -0.0676] | 0.000887 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.4574 | 0.6556 | -0.1981 | [-0.2472, -0.1514] | 4.6e-07 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.4949 | 0.8995 | -0.4046 | [-0.4532, -0.3551] | 6e-08 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.3912 | 0.4931 | -0.1019 | [-0.1185, -0.0861] | 7.78e-08 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.4949 | 0.6542 | -0.1593 | [-0.1861, -0.1343] | 5.77e-08 |
| map_accuracy | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.6435 | 0.8977 | -0.2542 | [-0.2921, -0.2185] | 5.77e-08 |
| log_loss | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.5067 | 1.5206 | -0.0139 | [-0.0246, -0.0045] | 0.162 |
| log_loss | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.1648 | 1.1951 | -0.0303 | [-0.0443, -0.0177] | 0.00126 |
| log_loss | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6816 | 0.7138 | -0.0322 | [-0.0476, -0.0185] | 0.00219 |
| log_loss | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 1.5651 | 1.5738 | -0.0087 | [-0.0153, -0.0028] | 0.127 |
| log_loss | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 1.2052 | 1.2229 | -0.0177 | [-0.0279, -0.0088] | 0.0105 |
| log_loss | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6701 | 0.6993 | -0.0292 | [-0.0432, -0.0165] | 0.00283 |
| log_loss | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.5014 | 1.5206 | -0.0192 | [-0.0326, -0.0073] | 0.0532 |
| log_loss | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.1737 | 1.1951 | -0.0214 | [-0.0424, -0.0002] | 0.127 |
| log_loss | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6813 | 0.7138 | -0.0325 | [-0.0550, -0.0098] | 0.0791 |
| log_loss | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 1.5644 | 1.5738 | -0.0095 | [-0.0181, -0.0003] | 0.119 |
| log_loss | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 1.2040 | 1.2229 | -0.0189 | [-0.0314, -0.0065] | 0.0392 |
| log_loss | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6678 | 0.6993 | -0.0315 | [-0.0529, -0.0100] | 0.0534 |
| log_loss | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.5200 | 1.5206 | -0.0006 | [-0.0087, +0.0073] | 1 |
| log_loss | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.1933 | 1.1951 | -0.0019 | [-0.0169, +0.0134] | 1 |
| log_loss | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.7165 | 0.7138 | +0.0027 | [-0.0115, +0.0173] | 1 |
| log_loss | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 1.5785 | 1.5738 | +0.0047 | [-0.0027, +0.0125] | 1 |
| log_loss | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 1.2266 | 1.2229 | +0.0037 | [-0.0042, +0.0121] | 1 |
| log_loss | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.7013 | 0.6993 | +0.0021 | [-0.0114, +0.0159] | 1 |
| log_loss | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.7122 | 1.5206 | +0.1916 | [+0.1727, +0.2092] | 3.86e-08 |
| log_loss | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 1.4679 | 1.1951 | +0.2727 | [+0.2441, +0.3005] | 3.86e-08 |
| log_loss | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 1.0901 | 0.7138 | +0.3763 | [+0.3398, +0.4108] | 2.56e-13 |
| log_loss | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 1.7227 | 1.5738 | +0.1489 | [+0.1341, +0.1630] | 3.84e-13 |
| log_loss | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 1.4636 | 1.2229 | +0.2407 | [+0.2191, +0.2606] | 3.84e-13 |
| log_loss | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 1.0788 | 0.6993 | +0.3795 | [+0.3454, +0.4117] | 2.56e-13 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.6325 | 1.5206 | +0.1120 | [+0.0953, +0.1273] | 7.07e-08 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 1.4121 | 1.1951 | +0.2169 | [+0.1923, +0.2410] | 3.86e-08 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 1.0481 | 0.7138 | +0.3343 | [+0.3020, +0.3663] | 2.56e-13 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 1.7021 | 1.5738 | +0.1283 | [+0.1132, +0.1434] | 3.94e-08 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 1.4379 | 1.2229 | +0.2150 | [+0.1917, +0.2377] | 3.84e-13 |
| log_loss | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 1.0494 | 0.6993 | +0.3502 | [+0.3206, +0.3794] | 2.56e-13 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.9661 | 1.5206 | +0.4455 | [+0.4103, +0.4803] | 2.56e-13 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 1.9053 | 1.1951 | +0.7102 | [+0.6691, +0.7508] | 3.86e-08 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 1.8642 | 0.7138 | +1.1504 | [+1.1028, +1.1970] | 2.56e-13 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 1.9792 | 1.5738 | +0.4053 | [+0.3846, +0.4257] | 2.56e-13 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 1.8855 | 1.2229 | +0.6626 | [+0.6280, +0.6970] | 2.56e-13 |
| log_loss | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 1.7477 | 0.6993 | +1.0484 | [+1.0021, +1.0936] | 2.56e-13 |
| regret | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.1711 | 0.1691 | +0.0020 | [-0.0008, +0.0059] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.1102 | 0.1116 | -0.0014 | [-0.0060, +0.0025] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.0224 | 0.0239 | -0.0015 | [-0.0049, +0.0019] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.2051 | 0.2057 | -0.0006 | [-0.0039, +0.0029] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.1056 | 0.1011 | +0.0045 | [-0.0003, +0.0105] | 1 |
| regret | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.0344 | 0.0352 | -0.0008 | [-0.0043, +0.0028] | 1 |
| regret | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.1704 | 0.1691 | +0.0012 | [-0.0020, +0.0054] | 1 |
| regret | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.1083 | 0.1116 | -0.0032 | [-0.0094, +0.0017] | 1 |
| regret | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.0273 | 0.0239 | +0.0034 | [-0.0022, +0.0111] | 1 |
| regret | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.2045 | 0.2057 | -0.0012 | [-0.0037, +0.0011] | 1 |
| regret | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.1043 | 0.1011 | +0.0032 | [-0.0017, +0.0086] | 1 |
| regret | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.0383 | 0.0352 | +0.0031 | [-0.0014, +0.0077] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.1724 | 0.1691 | +0.0032 | [-0.0005, +0.0080] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.1102 | 0.1116 | -0.0014 | [-0.0059, +0.0032] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.0245 | 0.0239 | +0.0006 | [-0.0066, +0.0086] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.2040 | 0.2057 | -0.0017 | [-0.0066, +0.0042] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.1039 | 0.1011 | +0.0028 | [-0.0035, +0.0099] | 1 |
| regret | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.0355 | 0.0352 | +0.0003 | [-0.0062, +0.0082] | 1 |
| regret | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.1722 | 0.1691 | +0.0031 | [-0.0114, +0.0193] | 1 |
| regret | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.1202 | 0.1116 | +0.0086 | [-0.0026, +0.0224] | 1 |
| regret | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.0392 | 0.0239 | +0.0153 | [+0.0039, +0.0272] | 0.445 |
| regret | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.2088 | 0.2057 | +0.0031 | [-0.0025, +0.0100] | 1 |
| regret | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.1225 | 0.1011 | +0.0215 | [+0.0094, +0.0338] | 0.0135 |
| regret | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.0475 | 0.0352 | +0.0123 | [+0.0015, +0.0238] | 1 |
| regret | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.1698 | 0.1691 | +0.0006 | [-0.0074, +0.0082] | 1 |
| regret | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.1191 | 0.1116 | +0.0076 | [-0.0031, +0.0204] | 1 |
| regret | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.0356 | 0.0239 | +0.0117 | [+0.0003, +0.0245] | 0.992 |
| regret | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.2128 | 0.2057 | +0.0071 | [-0.0029, +0.0177] | 1 |
| regret | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.1222 | 0.1011 | +0.0211 | [+0.0085, +0.0344] | 0.0661 |
| regret | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.0415 | 0.0352 | +0.0063 | [-0.0015, +0.0148] | 1 |
| regret | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.2557 | 0.1691 | +0.0866 | [+0.0465, +0.1267] | 0.00522 |
| regret | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.2148 | 0.1116 | +0.1032 | [+0.0648, +0.1407] | 0.000824 |
| regret | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.1894 | 0.0239 | +0.1654 | [+0.1316, +0.2020] | 2.4e-07 |
| regret | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.2594 | 0.2057 | +0.0537 | [+0.0310, +0.0767] | 0.00386 |
| regret | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.1957 | 0.1011 | +0.0946 | [+0.0733, +0.1165] | 5.06e-07 |
| regret | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.1238 | 0.0352 | +0.0886 | [+0.0677, +0.1105] | 4.89e-07 |
| hit@1 | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.0958 | 0.1042 | -0.0083 | [-0.0208, +0.0000] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.2250 | 0.2208 | +0.0042 | [-0.0125, +0.0250] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.7083 | 0.7125 | -0.0042 | [-0.0417, +0.0292] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.1042 | 0.1042 | +0.0000 | [-0.0125, +0.0125] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.3042 | 0.3042 | +0.0000 | [-0.0125, +0.0125] | 1 |
| hit@1 | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.5833 | 0.5750 | +0.0083 | [-0.0250, +0.0417] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.0958 | 0.1042 | -0.0083 | [-0.0208, +0.0000] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.2250 | 0.2208 | +0.0042 | [+0.0000, +0.0125] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6833 | 0.7125 | -0.0292 | [-0.0875, +0.0125] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.1083 | 0.1042 | +0.0042 | [+0.0000, +0.0125] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.2917 | 0.3042 | -0.0125 | [-0.0292, +0.0000] | 1 |
| hit@1 | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.5625 | 0.5750 | -0.0125 | [-0.0458, +0.0167] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 0.0958 | 0.1042 | -0.0083 | [-0.0208, +0.0000] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 0.2208 | 0.2208 | +0.0000 | [-0.0125, +0.0125] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.6958 | 0.7125 | -0.0167 | [-0.0708, +0.0375] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.1042 | 0.1042 | +0.0000 | [-0.0208, +0.0167] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.3042 | 0.3042 | +0.0000 | [-0.0167, +0.0167] | 1 |
| hit@1 | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.5750 | 0.5750 | +0.0000 | [-0.0417, +0.0375] | 1 |
| hit@1 | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.1042 | 0.1042 | +0.0000 | [+0.0000, +0.0000] | 1 |
| hit@1 | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.2042 | 0.2208 | -0.0167 | [-0.0667, +0.0167] | 1 |
| hit@1 | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.6000 | 0.7125 | -0.1125 | [-0.2125, -0.0125] | 0.866 |
| hit@1 | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.1000 | 0.1042 | -0.0042 | [-0.0167, +0.0083] | 1 |
| hit@1 | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.2542 | 0.3042 | -0.0500 | [-0.1083, +0.0042] | 1 |
| hit@1 | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.5292 | 0.5750 | -0.0458 | [-0.1167, +0.0250] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.1000 | 0.1042 | -0.0042 | [-0.0125, +0.0000] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.1833 | 0.2208 | -0.0375 | [-0.1042, +0.0125] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.5958 | 0.7125 | -0.1167 | [-0.2250, -0.0208] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.0833 | 0.1042 | -0.0208 | [-0.0500, +0.0125] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.2833 | 0.3042 | -0.0208 | [-0.0583, +0.0125] | 1 |
| hit@1 | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.5542 | 0.5750 | -0.0208 | [-0.0875, +0.0417] | 1 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 0.0833 | 0.1042 | -0.0208 | [-0.1250, +0.0833] | 1 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.1125 | 0.2208 | -0.1083 | [-0.2458, +0.0292] | 1 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.1750 | 0.7125 | -0.5375 | [-0.6750, -0.4000] | 2.1e-05 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.0667 | 0.1042 | -0.0375 | [-0.0833, +0.0167] | 1 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.1208 | 0.3042 | -0.1833 | [-0.2708, -0.0958] | 0.00947 |
| hit@1 | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.2708 | 0.5750 | -0.3042 | [-0.4083, -0.2000] | 0.000265 |
| violations | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.1417 | 1.1500 | -0.0083 | [-0.0208, +0.0000] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2208 | 1.2375 | -0.0167 | [-0.0667, +0.0250] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.9500 | 0.9375 | +0.0125 | [-0.0292, +0.0542] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.7417 | 0.7417 | +0.0000 | [-0.0125, +0.0125] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.7458 | 0.7250 | +0.0208 | [-0.0083, +0.0542] | 1 |
| violations | elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.7167 | 0.7167 | -0.0000 | [-0.0333, +0.0333] | 1 |
| violations | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.1417 | 1.1500 | -0.0083 | [-0.0208, +0.0000] | 1 |
| violations | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2250 | 1.2375 | -0.0125 | [-0.0625, +0.0250] | 1 |
| violations | elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.9375 | 0.9375 | +0.0000 | [-0.0500, +0.0500] | 1 |
| violations | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.7417 | 0.7417 | +0.0000 | [-0.0125, +0.0125] | 1 |
| violations | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.7500 | 0.7250 | +0.0250 | [-0.0042, +0.0625] | 1 |
| violations | elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.6917 | 0.7167 | -0.0250 | [-0.0625, +0.0083] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 1.1417 | 1.1500 | -0.0083 | [-0.0208, +0.0000] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 1.2167 | 1.2375 | -0.0208 | [-0.0792, +0.0375] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 0.9833 | 0.9375 | +0.0458 | [+0.0000, +0.0958] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 0.7500 | 0.7417 | +0.0083 | [-0.0083, +0.0250] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 0.7333 | 0.7250 | +0.0083 | [-0.0250, +0.0417] | 1 |
| violations | elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 0.7292 | 0.7167 | +0.0125 | [-0.0292, +0.0542] | 1 |
| violations | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.1125 | 1.1500 | -0.0375 | [-0.1000, +0.0125] | 1 |
| violations | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 1.2708 | 1.2375 | +0.0333 | [-0.0333, +0.1000] | 1 |
| violations | elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.8625 | 0.9375 | -0.0750 | [-0.2000, +0.0417] | 1 |
| violations | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.8208 | 0.7417 | +0.0792 | [+0.0292, +0.1333] | 0.151 |
| violations | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.7375 | 0.7250 | +0.0125 | [-0.0625, +0.1000] | 1 |
| violations | elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.5792 | 0.7167 | -0.1375 | [-0.2583, -0.0375] | 0.624 |
| violations | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.1542 | 1.1500 | +0.0042 | [-0.0293, +0.0375] | 1 |
| violations | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 1.2625 | 1.2375 | +0.0250 | [-0.0208, +0.0833] | 1 |
| violations | elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 1.0667 | 0.9375 | +0.1292 | [-0.0125, +0.2792] | 1 |
| violations | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 0.9000 | 0.7417 | +0.1583 | [+0.1042, +0.2167] | 0.00101 |
| violations | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 0.8333 | 0.7250 | +0.1083 | [+0.0292, +0.2000] | 0.624 |
| violations | elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.7250 | 0.7167 | +0.0083 | [-0.0833, +0.0917] | 1 |
| violations | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 1.1042 | 1.1500 | -0.0458 | [-0.2125, +0.1167] | 1 |
| violations | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 0.9792 | 1.2375 | -0.2583 | [-0.4125, -0.1125] | 0.0706 |
| violations | elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 0.7750 | 0.9375 | -0.1625 | [-0.3542, +0.0292] | 1 |
| violations | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 1.0958 | 0.7417 | +0.3542 | [+0.2708, +0.4375] | 3.04e-06 |
| violations | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 1.0875 | 0.7250 | +0.3625 | [+0.2125, +0.5125] | 0.00217 |
| violations | elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 0.9375 | 0.7167 | +0.2208 | [+0.0958, +0.3542] | 0.0706 |

### Episodes that asked the teacher's questions in the teacher's order

| model | arm | K | episodes | same |
|---|---|---:|---:|---:|
| elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=eig,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-1.7b-sft | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=eig,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-4b-sft | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=eig,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-upd-0.6b-sft | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=eig,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-qwen3-4b-zs | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=eig,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-qwen3-1.7b-zs | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
| elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 3 | 240 | 0.000 |
| elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 5 | 240 | 0.000 |
| elicitor-qwen3-0.6b-zs | sel=eig,upd=llm,phr=tpl | 8 | 240 | 0.000 |
| elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 3 | 240 | 1.000 |
| elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 5 | 240 | 1.000 |
| elicitor-qwen3-0.6b-zs | sel=random,upd=llm,phr=tpl | 8 | 240 | 1.000 |
