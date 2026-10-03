# EEG denominator sensitivity

Run the standalone teacher command with a local copy of
`configs/eeg_denominator_sensitivity.example.json`:

```powershell
python scripts/run_teacher_analysis.py eeg-denominator-sensitivity --config configs/eeg_denominator_sensitivity.local.json --dry-run
python scripts/run_teacher_analysis.py eeg-denominator-sensitivity --config configs/eeg_denominator_sensitivity.local.json --outdir D:/study-local/new-run
```

The command requires a committed, clean repository and a new output directory.
Read-only preflight checks the archived four-window sample, design, epoch fields,
waveform pairs, launchers and R dependencies. MATLAB checks the actual waveform,
ROI and spectrum validity during execution. EEGLAB does not run: existing SET
metadata and single-precision little-endian FDT data are read directly, with no
filtering, ICA, reference change or QC recomputation.
R receives exact UTF-8 input copies under a temporary ASCII working path to
avoid legacy Windows command-line filename conversion; Python copies the
unchanged results back into the requested Unicode output directory.

## Three versions

A is the archived 0805 model input. B is reproduced 1–45 Hz power from the current
preprocessed waveforms. C uses exactly B's spectra and numerators with a 1–40 Hz
denominator. B→C isolates the denominator effect. A→B quantifies reproduction
drift; A→C checks manuscript values. Archive keys, covariates, predecessors,
inclusion, QC and epoch bounds are frozen; no invalid row is silently discarded.
The archive's missing Complexity is restored to C0 only with archived Cond=C0
evidence. PreviousComplexity is never rebuilt from the retained subset.
Its missing C0 encoding is decoded only where archived PreviousWWR identifies an
existing predecessor. Historical auxiliary bands/ratios outside the requested
18 outcomes remain archived values and are not modeled; bare theta/alpha/beta
absolute aliases are updated together with their explicit absolute columns.

ROI averaging precedes pwelch. Numeric-window MATLAB pwelch, inclusive sample
endpoints, rounded onset removal, original frequency masks and trapz are retained.
Each participant MAT cache stores complete f/pxx and spectrum parameters. Cache
identity includes source SHA256, trial bounds, ROI, PSD parameters and MATLAB
implementation hashes. Every artifact hash must match on reuse; incomplete or
modified caches are refused. Relative and historical QC denominators are separate.
The 40–45 fraction differs from (P1–45−P1–40)/P1–45 because the discrete masks leave
a trapezoid between their boundary bins. Both quantities are reported per spectrum.

## Statistical contract

ML random-intercept LMM and participant-clustered CR2 coefficient inference remain
unchanged. Core Model0, PreviousScene and Block1 are retained. PreviousScene uses
PreviousWWR + PreviousComplexity, reflecting actual historical execution; the
manuscript interaction description is recorded as inconsistent.

Factor-level inference is explicitly reimplemented, not recovered historical code:
six equal-weight marginal contrasts with CR2 covariance and HTZ Wald tests. WWR
main effects average over Complexity and ExperienceGroup; the latter main effects
average over all three WWR levels and both levels of the other binary factor.
Contrast matrices and denominator degrees of freedom are saved. Multi-df F tests
have no fabricated beta or confidence interval. Single-df contrasts include
unadjusted 95% intervals. Near-zero beta percentage changes are left unavailable.
The percentage-change guard is |baseline beta|<1e-4. Both alpha and theta
position effects are included in the manuscript review; borderline joint-q
flips and nonsignificant near-zero direction changes are explicitly reported.

Within-window and joint BH families are separate: coefficient core relative144,
expanded relative324, expanded absolute324; factor96/216/216; temporal72/72;
previous core48. Failed inference leaves the complete family present and final q
unavailable. The A version must pass 240 archived core Model1 coefficients at
beta1e-8 and p1e-6, plus the independently checked factor table at p/q1e-6.
B/C absolute input and output equality is asserted before reuse.

CSV comparisons include all prescribed families, numerical changes, raw p,
window q, joint q, direction and significance flips. A Chinese report maps the
manuscript claims to these tests. Status complete means the analysis is complete,
not that conclusions must remain unchanged. Local research results and PSD caches
are never Git content. The dedicated sensitivity result pointer is distinct from
the existing multimodal result entry.

`all_model_coefficients_unadjusted` also pairs intercepts, covariates and core
Model0/Block1/PreviousScene coefficients without inventing additional BH families.
Their q fields are deliberately unavailable. The companion
`scripts/export_eeg_denominator_workbook.mjs` exports typed comparison tables via
the bundled Artifact Tool; statistics remain owned by the R/Python pipeline.
Generate its typed input with `python scripts/prepare_eeg_denominator_workbook.py
<completed-run-directory>`. Workbook tables contain aggregate model results and
diagnostics, without trial-level participant names. Preserve both raw p and both
q scopes; the main sheet judges direction and joint-BH significance per claim.

# 全流程正式结果更新

在 `all-results` 配置中加入与独立命令相同的 `denominator_sensitivity` 配置，
会在 `10_eeg_denominator_sensitivity` 完成 A/B/C 重分析后再验收和推广。
`--promote` 同时备份并更新 `12_teacher_analysis`，总报告、完成矩阵和
`realdata_run_summary.json` 纳入分母敏感性及执行提交。主流程历史 1–45 Hz
结果保留，1–40 Hz 为独立敏感性版本。使用现有 PSD 缓存前验证完整身份与文件哈希。

全流程 R 阶段使用 ASCII 临时目录运行原脚本和逐字节复制的输入，再通过 Python
回写中文路径；R 失败时不推广部分结果。运行日志保存在各阶段目录。

`--reuse-valid` 还会核对已登记 R 输出的哈希、全部输入字段与行、统计脚本及
标量参数。仅输入读取方式改变或历史空白 C0 显式改名、数值仅有不超过
`1e-12` 的写出舍入差异时，允许复用同一统计实现的旧结果，包括5000次
bootstrap。复用证明保存在各阶段的 `*_reuse.json`；条件不满足则重跑。

老师交付使用 `scripts/publish_latest_teacher_results.py` 的 `prepare`、`index`、
`finalize` 三阶段，只接受已完成并推广的全流程。`prepare` 生成 Word 简报、
机器可读关键对照及工作簿 typed JSON；`scripts/export_teacher_workbook.mjs`
使用外部 bundled runtime 将 JSON 导出 Excel。完成文档渲染检查后，`index`
生成交付内容与哈希索引的 typed JSON，导出索引 Excel；`finalize` 核对索引
哈希、生成并验证 ZIP，再更新老师目录和正式结果目录的最新交付入口。
已发布目录不允许刷新；未发布草稿可用 `--refresh-draft` 更新。

仅 R 缓存检查函数发生变化时，`scripts/revalidate_teacher_transport.py` 可
核验并复用已完成阶段：它要求全部其他方法依赖无变化，R 调用与输入暂存
的 AST 相同，原输入哈希仍一致；保留原执行提交，另记本次验证提交及输出
哈希。数据准备、推断代码或 R 调用改变时拒绝这种复用。
