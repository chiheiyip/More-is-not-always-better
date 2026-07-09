# Teacher Data Brief

这份摘要只把数据结果摊开，供老师判断写法；不是最终论文结论。

## 样本和口径

- 参与者：56
- 场景试次：总计 672，主分析保留 453，排除 219
- AOI 展开行：1130。这个数字不是场景试次数，不能和 453 个保留场景试次混用。

主要排除原因：
- bad_eeg_quality: 201
- duration_mismatch: 23
- missing_questionnaire: 0
- missing_eye: 0
- missing_eeg: 0
- scene_count_mismatch: 0

## 问卷结果

- S1 按 WWR：最高为 WWR=15，均值 5.54；最低为 WWR=45，均值 5.26；差值约 0.277。
- S2 按 WWR：最高为 WWR=15，均值 5.78；最低为 WWR=45，均值 5.33；差值约 0.446。
- S3 按 WWR：最高为 WWR=15，均值 5.4；最低为 WWR=75，均值 5.08；差值约 0.321。
- S4 按 WWR：最高为 WWR=15，均值 5.49；最低为 WWR=45，均值 5；差值约 0.487。
- S5 按 WWR：最高为 WWR=15，均值 6.78；最低为 WWR=45，均值 6.23；差值约 0.554。
- S1 按 Complexity：最高为 Complexity=1，均值 5.51；最低为 Complexity=0，均值 5.29；差值约 0.22。
- S2 按 Complexity：最高为 Complexity=1，均值 5.54；最低为 Complexity=0，均值 5.48；差值约 0.0685。
- S3 按 Complexity：最高为 Complexity=1，均值 5.31；最低为 Complexity=0，均值 5.26；差值约 0.0476。
- S4 按 Complexity：最高为 Complexity=1，均值 5.33；最低为 Complexity=0，均值 5.23；差值约 0.101。
- S5 按 Complexity：最高为 Complexity=1，均值 6.55；最低为 Complexity=0，均值 6.51；差值约 0.0357。

## 眼动结果

- visited 按 WWR：最高为 WWR=75，均值 0.941；最低为 WWR=15，均值 0.787；差值约 0.154。
- FCR 按 WWR：最高为 WWR=75，均值 0.333；最低为 WWR=15，均值 0.24；差值约 0.0924。
- TFD_ms 按 WWR：最高为 WWR=75，均值 8.22e+03；最低为 WWR=45，均值 7.28e+03；差值约 949。
- TTFF_ms 按 WWR：最高为 WWR=15，均值 8.92e+03；最低为 WWR=75，均值 5.63e+03；差值约 3.29e+03。
- attention_share 按 WWR：最高为 WWR=75，均值 0.175；最低为 WWR=45，均值 0.152；差值约 0.0235。
- visited 按 Complexity：最高为 Complexity=1，均值 0.88；最低为 Complexity=0，均值 0.866；差值约 0.0139。
- FCR 按 Complexity：最高为 Complexity=0，均值 0.325；最低为 Complexity=1，均值 0.257；差值约 0.0685。
- TFD_ms 按 Complexity：最高为 Complexity=0，均值 9.24e+03；最低为 Complexity=1，均值 6.84e+03；差值约 2.4e+03。
- TTFF_ms 按 Complexity：最高为 Complexity=1，均值 7.61e+03；最低为 Complexity=0，均值 6.88e+03；差值约 737。
- attention_share 按 Complexity：最高为 Complexity=0，均值 0.192；最低为 Complexity=1，均值 0.143；差值约 0.0487。

## EEG 结果

- F_theta 按 WWR：最高为 WWR=75，均值 49.2；最低为 WWR=15，均值 20.3；差值约 28.9。
- O_theta 按 WWR：最高为 WWR=75，均值 32.4；最低为 WWR=15，均值 28.8；差值约 3.56。
- O_alpha 按 WWR：最高为 WWR=45，均值 16.4；最低为 WWR=15，均值 13；差值约 3.43。
- F_theta 按 Complexity：最高为 Complexity=0，均值 38.9；最低为 Complexity=1，均值 32.7；差值约 6.16。
- O_theta 按 Complexity：最高为 Complexity=0，均值 40.8；最低为 Complexity=1，均值 20.5；差值约 20.3。
- O_alpha 按 Complexity：最高为 Complexity=0，均值 18；最低为 Complexity=1，均值 11.2；差值约 6.76。

## 显著性结果怎么看

- p<0.05 的模型/对比项：71 行。
- OLS fallback 行：81 行。
- 带模型警告或不稳定标记行：151 行。
- 显著性只说明该比较或模型项显著；是否写成机制、最优或规律，需要老师结合实验设计和模型稳定性判断。
