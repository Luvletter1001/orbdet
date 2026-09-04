# SCQO Plan A HRSC 只读证据审计

本报告只读取冻结 checkpoint 和 HRSC validation；没有训练、调参或测试集评估。

| run | Gate B | eligible | AUROC | baseline | gain | e2 median | e2 P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| godc_seed3407_best | FAIL | 444 | 0.5559 | 0.4485 | 0.1074 | 82.68 | 89.43 |
| v02_seed3407_best | FAIL | 442 | 0.5447 | 0.4063 | 0.1385 | 85.34 | 89.64 |

Gate B 要求：combined AUROC ≥ 0.65、相对最佳能量/方差单变量基线提高 ≥ 0.05，并且低、中、高三个纹理桶的 AUROC 均 > 0.5。

PASS 只授权继续讨论 Plan B；FAIL/INSUFFICIENT 均不得启动 E4。
