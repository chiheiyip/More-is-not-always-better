# Interpretation Reference

本文件是参考性解释，不替代老师基于数据的最终判断。

## 可以直接从数据表判断的事

- 每个指标在 WWR、Complexity、ExperienceGroup 下的均值、差异方向和样本口径。
- 模型项或 planned contrast 是否达到 p<0.05。
- 结果是否来自 scene trial、AOI-expanded row 或 smoke run。

## 需要谨慎的事

- WWR 只有 15、45、75 三档；即使 45% 显著，也只能说在本实验测试的三档中表现更优，不能直接说找到了连续意义上的最优 WWR。
- EEG 解释必须结合 EEG QC、同步 QC 和问卷/眼动收敛，不能单靠 EEG 模型项写认知机制。
- ExperienceGroup 只有交互项稳定时才支持“调节效应”；主效应更适合写成经验组差异。
- AOI-expanded row 不是 scene trial。眼动模型和描述统计必须标明数据粒度。
- 小样本 smoke run 只验证流程，不更新论文最终实验结果。

## WWR 显著性参考

- WWR 相关 p<0.05 行数：20。请优先回看 `experiment_significance_results.csv` 的 estimate、CI、model_type 和 warning_flag。
