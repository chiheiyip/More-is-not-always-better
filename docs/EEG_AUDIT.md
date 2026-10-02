# EEG独立代码验证

`eeg-audit` 使用已提取的真实或模拟频带功率验证代码，不重新提取PSD，不提升为正式结果。原有命令和历史结果不被覆盖。新目录必须不存在；不复用缓存。代码、配置或输入改变后必须使用新目录。

## 使用

根据 `configs/eeg_audit.example.json` 填写实际输入及R启动器。相对路径以配置所在目录解析。输入是完整四窗口试次表、参与者登记表和已核验的预处理说明；只有已核实来源才将 `preprocessing_confirmed` 设为 true。

```powershell
python scripts/run_teacher_analysis.py eeg-audit --config validation_config.json --outdir outputs/eeg_validation/check_001 --dry-run
python scripts/run_teacher_analysis.py eeg-audit --config validation_config.json --outdir outputs/eeg_validation/check_001
```

`--dry-run` 读取输入并验证共同样本，不创建输出或拟合模型。正式执行退出码0表示本次验证完整通过，3表示模型/检验族存在问题，2表示输入或运行约定不满足；外部R执行错误返回非零。`--skip-r`不能产生成功验证。程序不执行合并、推送或结果推广。

## 分析约定

四窗口0/5/10/15秒同等平行，10秒只作为兼容参照。额F、顶P、枕O分别有theta/alpha/beta，9个relative与9个log10 absolute指标形成72个Model1。原4核心relative指标为O_theta、F_theta、O_alpha、O_beta，另有48个Model0/PreviousScene/Block1补充模型，单独计数。

Model1固定使用 `WWR*Complexity + WWR*ExperienceGroup + Complexity*ExperienceGroup + Gender + Block + PositionWithinBlockCentered + OrderGroup + (1|Participant)`，ML拟合、参与者聚类CR2。Block为数值1/2；轮内位置减3.5。参照为WWR15、C0、High、Female、new order2；全部因子水平写入拟合前保存的分析约定，不随样本自动改变。

在完整试次顺序上生成同轮上一场景，然后应用既有参与者与场景QC。缺少真实相邻位置时不跨缺口连接。四窗口、18个结局、固定协变量共同完整的试次构成Model1共同样本。absolute≤0及非有限功率显式排除，不添加常数。PreviousScene仅使用position>1且上一场景变量完整的试次；Block1只用第一轮。每个模型记录实际样本键、人数、试次数和键哈希，不硬编码历史样本量。

CR2系数、SE、Satterthwaite自由度、双侧原始p与95%置信区间来自同一聚类协方差。置信区间不作BH调整。模型警告、优化器返回码、奇异拟合、秩亏和CR2状态分别记录；本流程保守地仅在拟合收敛、非奇异、满秩且CR2有效时发布最终校正结论，问题模型的原始系数仍保留。

| 检验族 | 每窗口 | 联合四窗口 |
| --- | ---: | ---: |
| relative时序：9指标×Block/position | 18 | 72 |
| log10 absolute时序 | 18 | 72 |
| 核心relative条件效应 | 36 | 144 |
| 扩展relative条件效应 | 81 | 324 |
| 扩展log10 absolute条件效应 | 81 | 324 |
| 核心relative上一场景 | 12 | 48 |

条件效应的9个系数：WWR45/75、ComplexityC1、ExperienceGroupLow、WWR×Complexity两项、WWR×Experience两项、Complexity×Experience一项。上一场景为PreviousWWR45/75及PreviousComplexityC1。核心和扩展是不同检验族，各自重新计算BH；不继承历史GEE的117/468项q值。主判断使用各自`joint_q < 0.05`，`within_q`明确单列。成员缺失或无效时不缩小校正范围，受影响的窗口或联合检验族q留空，结论为未解决。并未将两个时序72项族再合并为144项族。

这些规则是当前代码验证约定，不证明历史预注册或过去144项primary family。历史决策仍需历史证据。分母1–45 Hz仅对已核验的旧输入来源作一致性检查；本命令不据配置文字重新计算功率。

## 输出与追溯

顶层README从机器表自动生成。`coefficients.csv`保存完整系数，`diagnostics.csv`和`model_samples.csv`保存逐模型诊断/样本，`family_coefficients.csv`列出所有预期假设及两类q，`family_status.csv`记录完整性。alpha/theta/上一场景检查均由同一表筛选，不硬编码目标结论；theta范围明确覆盖三个ROI的Block系数。

`analysis_contract.json`在模型拟合前记录输入哈希、实际配置、方法文件哈希、种子和设计。`code_snapshot`保留执行源码，Git状态与已跟踪差异单独保存，R日志及sessionInfo逐窗口保存。`output_hashes.json`记录本次产物。内部执行记录保留可复现所需的原始路径；阅读报告只使用相对链接。源码和日志不做措辞替换。

测试：`python -m pytest tests/test_eeg_audit.py`。R集成测试使用模拟数据，比较CR2区间与t分布公式、p值与双侧t检验，并将BH结果与R `p.adjust`对照。没有便携R时该项显式跳过，不能宣称R验证已通过。

## 真实数据验证封装

`scripts/validate_eeg_audit.py`在独立目录执行上述流程，随后与历史核心Model1比较。参数为`--config`、`--outdir`、`--historical-dir`（包含历史`onset_window_models`的阶段目录），以及可重复的`--protect`（需要保持不变的既有目录）。输出目录不得位于任何受保护目录之内。

封装逐窗口、结局和系数匹配历史表，不把其他窗口的结果重复计作缺失项。历史空复杂度只有在原`Cond`明确为C0时才在比较副本中规范化，不修改历史文件。比较同时检查样本键及规范化后的设计/核心功率，数值容差为β绝对差`1e-8`、原始p绝对差`1e-6`；超限或缺项标记为需调查，不通过调整模型消除差异。

`historical_comparison_summary.json`与两张逐项比较CSV记录对照结果。`protected_directory_check.json`比较受保护目录全部文件的路径、大小及纳秒级修改时间；直接分析输入另外校验SHA-256。执行发生异常时仍写保护检查。所有这些产物只用于代码验收，不替代历史结果或论文定稿依据。
