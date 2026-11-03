"""Build a factual report from saved outputs; no detector fitting or reruns."""
import json
from pathlib import Path
import pandas as pd


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def rate(value,interval):
    if value is None:
        return '不适用'
    return f'{100*value:.3f}% [{100*interval[0]:.3f}%, {100*interval[1]:.3f}%]'


def number(value):
    return '未检出' if value is None else f'{value:.3f}'


def main():
    base=Path('reports/stage3')
    experiments=[json.loads((directory/'results.json').read_text()) for directory in (base,base/'misspecification',base/'boundary')]
    rows=[row for experiment in experiments for row in experiment['summary']]
    pd.DataFrame(rows).to_csv(base/'all_summary.csv',index=False)
    config=experiments[0]['config']
    matched=json.loads((base/'matched_prediction.json').read_text())['summary']
    null_rows=[]
    for row in rows:
        if row['scenario'].startswith('normal_') and row['method']=='conditional_seq_fixed':
            null_rows.append([row['scenario'],row['alpha'],rate(row['false_alarm_rate'],row['false_alarm_ci95']),
                              f"{100*row['benign_envelope_violation_rate']:.1f}%"])
    attack_rows=[]
    for row in rows:
        if not row['scenario'].startswith('normal_') and row['method']=='conditional_seq_fixed' and row['alpha']==.05:
            attack_rows.append([row['scenario'],row['strength'],rate(row['false_alarm_rate'],row['false_alarm_ci95']),
                                rate(row['detection_rate'],row['detection_ci95']),number(row['mean_delay']),row['censored']])
    strong_rows=[]
    for row in rows:
        if row['strength']==1 and row['method']=='conditional_seq_fixed':
            strong_rows.append([row['scenario'],row['alpha'],rate(row['detection_rate'],row['detection_ci95']),
                               number(row['mean_delay']),number(row['restricted_mean_delay'])])
    bound_rows=[]
    for row in rows:
        if row['strength']==1 and row['alpha']==.05 and row['method'].startswith('conditional_seq_'):
            bound_rows.append([row['scenario'],row['method'].removeprefix('conditional_seq_'),
                               rate(row['detection_rate'],row['detection_ci95']),number(row['mean_delay'])])
    mse_rows=[[row['scenario'],f"{row['history_mse']:.8g}",f"{row['conditional_mse']:.8g}",
               f"{row['paired_improvement']:.8g} [{row['paired_ci95'][0]:.8g}, {row['paired_ci95'][1]:.8g}]"] for row in matched]
    aggregate_rows=[]
    for scenario in config['scenarios'][4:]:
        selected=[row for row in rows if row['scenario']==scenario and row['alpha']==.05 and row['strength']==1]
        detector=next(row for row in selected if row['method']=='conditional_seq_fixed')
        full=next(row for row in selected if row['method']=='full_seta_candidate')
        aggregate_rows.append([scenario,f"{detector['aggregation_mse']:.8g}",f"{full['aggregation_mse']:.8g}",
                               f"{100*(1-full['aggregation_mse']/detector['aggregation_mse']):.2f}%"])
    report='''# SETA-FL 第三阶段：统计理论与抗漂移验证报告

日期：2026-11-03。分支 `seta-stat-validation` 基于 `seta-math-fix`；未修改 main，未执行完整 CIFAR-10 或 FEMNIST 实验。

## 先回答五个问题

1. **序贯证据是否满足统计假设？** 指数构造在有界观测、可预测参数及正确条件均值包络下有完整超鞅证明。固定/滚动校准在本报告明确的稳定仿射、外生上下文、独立正常参考等前提下可分配校准失败预算。真实高维联邦更新尚未认证这些前提，不能宣称真实 FL 的严格误报率保证。
2. **正常动态漂移下是否控制误报？** 观测到的上下文解释漂移时，本次 10000 次重复没有误报。未观测突变漂移使固定校准误报 100%；未观测渐进漂移在 alpha=.05 时误报 82.32%。滚动参考在这些样本上未报警，但突变时包络也曾失效，不能据此宣称任意漂移保证。
3. **间歇性和隐蔽攻击能否检出？** 强间歇攻击可检出，但较弱间歇攻击和所有本次弱幅度隐蔽攻击均未检出。晚发攻击更难，alpha=.01 时强晚发攻击固定校准仅检出 59.98%。没有通用高检测率结论。
4. **条件预测是否有独立优势？** 在完全相同 2048 个正常参考样本的配对消融中，IID/固定 Non-IID 下无优势；上下文确实解释动态/渐进漂移时有明显的预测 MSE 优势。收益来自有效协变量，不是序贯检测本身，且依赖模型正确；不能推广为真实数据结论。
5. **是否值得进入正式 CIFAR-10/FEMNIST？** 值得继续做受控的小规模真实数据先导研究，暂不建议直接进入正式长训练/性能主张。先解决数据访问、真实条件校准、未知漂移和弱攻击功效，再决定扩大实验。FEMNIST 尚未加载或验证。

## 数学结论与前提

完整审查见 [理论文档](../../docs/STAGE3_THEORY.md)。使用 G_t=F_(t-1) 加上更新前已知的本轮上下文/参与信息；零假设是每个参与轮都有 |E[X_t|G_t]-muhat_t|<=b_t。

X∈[A,B] 时 Hoeffding 参数 sigma²=(B-A)²/4；当前 X∈[-1,1]，所以 sigma²=1。对固定正负下注 a：

\[
L_t^a=\\exp\\sum_{r\\le t}[a(X_r-\\widehat\\mu_r)-|a|b_r-a^2/2],
\\qquad E_t=\\tfrac1{10}\\sum_aL_t^a.
\]

条件 Hoeffding 引理与均值包络给出单步因子条件期望<=1，各 L 为非负超鞅，固定有限混合仍有效。Ville 控制 P(sup E>=1/alpha_e)<=alpha_e。每个总 alpha 预留 .1alpha 给校准失败，alpha_e=.9alpha；在校准前提下总误报界<=alpha_e+delta=alpha。真实 FL 的前提尚未验证。

预测均值、界、方向、下注与混合权重均需在当前更新前确定。不能由标准差、经验分位数、当轮测试标签、当前残差或未报警筛选宣称获得严格条件界。生产训练脚本依旧将其外部给定界标为假设。

## 三种校准及污染防护

- **固定历史校准**：在上下文 -1,+1 各 1024 个独立正常样本，两锚点同时 Hoeffding 界 epsilon_A=sqrt(2 log(4/delta)/n)。稳定仿射正常均值下，插值的全上下文误差<=epsilon_A。
- **滚动历史校准**：最近 1024 个独立验证参考（每轮 64 个，延迟一轮可用），对固定 240 个窗口分配联合失败预算；条件残差校正界为 epsilon_R+epsilon_A*mean|c_t-c_j|。不把滚动样本标准差当作证明。
- **保守不确定性界**：b=1+|muhat| 覆盖所有 [-1,1] 条件均值，确定性有效，但财富不增，检测力为零。其零报警不代表成功。

待测目标更新从不进入参考流；正常历史来自独立模拟的外部可信服务。真实系统是否拥有这样的服务是待解决前提。`ExternalCleanWindow` 拒绝未到验证时间或非显式参考对象；来源文本本身不是认证。

## 实验协议与可追溯性

核心协议先在提交 `4c7150e` 固定，未观测漂移负对照在 `39f38ef` 固定，边界零假设及同数据消融在 `68bd923` 固定。补充对照是在主实验后增加的反证分析，不是声称全部实验在首次结果前预注册。没有利用测试标签调整任何检测参数；所有强度和预算在对应运行前确定。

核心 8 场景：4 个正常场景各 10000 次，4 个攻击场景各 3 种强度、每强度 10000 次，共 16 个单元。再加 2 个未观测漂移负对照、1 个 Rademacher 边界零假设，合计 19 个单元、190000 条单元轨迹。每条 240 轮，alpha=.01/.05/.10 配对复用，7 种方法，共 3990000 行原始结果。跨强度使用共同随机数，不应把这些单元当作全部相互独立；每个单元内部有 10000 个独立重复。

另有 40000 条相同训练数据的预测消融记录。原始 CSV 以 `.csv.gz` 无损压缩保存，均可由 pandas 或 gzip 直接读取；JSON 记录源代码/配置哈希，`source_snapshot/` 保存对应运行代码。日志未隐藏失败。所有结果均为**有界标量特征统计模拟**，不是 CIFAR 模型准确率或真实联邦检测率。

正常 Non-IID 是不同重复中客户端均值不同、各自独立参考；不是同时模拟多个异质模型的真实训练。聚合代理使用一个权重 .2 的受监控客户端和四个正常同均值客户端。条件模型的仿射形式与数据生成机制一致，这是验证特例，不是现实适用性的证据。

## 误报率：固定校准，95% Wilson 区间

正常场景为 240 轮内至少一次报警。最后一列为使用已知生成机制**仅在评估时**诊断的均值包络违例率，不用于拟合。

'''
    report+=table(['场景','alpha','经验误报率 [95% CI]','包络违例率'],null_rows)
    report+='''

固定/滚动校准在解释漂移场景均为零误报；0/10000 的 Wilson 上界约 .0384%，不是总体概率为零。保守策略所有场景均不报警。未观察突变时滚动策略虽无报警，实际包络违例率为 100%，故不能根据小误报率反向证明假设成立。

## 攻击强度：固定校准，alpha=.05

持续/间歇/信誉积累攻击为 X_attacked=(1-2*strength)*X_clean；strength=1 是完整 Sign Flipping，.5 是归零，.25 是减半，不能都称为完整符号翻转。弱幅度攻击为 X_clean-.2*strength。间歇攻击每四轮前两轮生效；普通攻击第 41 轮开始，信誉积累场景前 160 轮表现正常、第 161 轮攻击。这里的信誉积累是晚发攻击行为，不是已实现独立动态信誉评分算法。

攻击前误报窗口分别为 40/160 轮；检测率分母排除提前报警者，不能和正常场景的 240 轮误报窗口混淆。延迟含攻击发生当轮。漏检单列，平均延迟只对检出者有定义。

'''
    report+=table(['场景','强度','攻击前误报率 [CI]','检测率 [CI]','平均延迟','漏检'],attack_rows)
    report+='\n\n## 三种 alpha：最大攻击强度、固定校准\n\n'
    report+=table(['场景','alpha','检测率 [95% CI]','检出者平均延迟','截断平均延迟'],strong_rows)
    report+='''

无检出时不能报告延迟为零。截断平均延迟把未检出者记为完整观察窗（常规 200 轮，晚发 80 轮）。较小 alpha 对晚发攻击更保守；攻击前财富衰减使 160 轮正常行为后的检测更慢。这不是降低阈值可以掩盖的问题。

## 校准方式的功效代价：最大强度、alpha=.05

'''
    report+=table(['场景','校准方式','检测率 [95% CI]','检出者平均延迟'],bound_rows)
    report+='''

保守界虽无条件覆盖，却没有检测力。滚动界为多个窗口预留预算，可能比固定界更宽，在当前正确条件模型下功效更低；不能把“滚动”直接等同于更优。

## 预测消融与独立优势

主实验比较 history_mean、rolling_mean、conditional、conditional+三种序贯界、full_seta_candidate。前三个是预测器，没有检测器，误报/检测率为“不适用”，没有伪造零检测率。

主实验条件模型使用两锚点共 2048 个样本，普通初始均值用 1024，样本量混杂不能用于宣称条件化独立优势。因此增加下表控制：**两者使用完全相同 2048 个锚点样本**，普通均值忽略上下文，条件预测使用上下文；没有序贯组件。95% 配对区间反映重复之间差异，不是模型正确性证明。

'''
    report+=table(['场景','普通均值 MSE','条件预测 MSE','配对 MSE 改善 [95% CI]'],mse_rows)
    report+='''

IID 和固定 Non-IID 的 c=0 时两预测完全相同，所以无独立优势。上下文解释漂移时，条件化显著改善预测。这是信息有用且模型正确时的构造性结果，不能宣称对未观测漂移或真实 FL 已成立。滚动方法还获得持续外部参考，信息预算不同，不应将其与静态方法当作完全等资源比较。

## “完整 SETA-FL”候选管线与聚合消融

目前研究方法尚未完整定义，不能把一个临时加权公式冒充最终算法。本阶段提供可运行的 **full_seta_candidate**：条件预测 + 固定校准序贯证据 + 隔离记录 + 报警锁存 + 报警客户端权重归零、其他样本数权重重新归一化。实际训练接口为 `--aggregation evidence-filter`；默认仍是 FedAvg。所有客户端被过滤时不更新全局模型。

该候选与 conditional_seq_fixed 的检测时间和检测率应完全相同；它只增加聚合动作。单元测试确认观测与决定发生在聚合前。以下只是标量聚合 MSE，不能解释成模型准确率。

'''
    report+=table(['场景（强度1，alpha=.05）','仅检测/FedAvg MSE','候选过滤 MSE','相对降低'],aggregate_rows)
    report+='''

弱攻击未报警，所以过滤没有改善。真实模型中的选择反馈会影响未来更新分布，可能使校准失效；当前代理实验没有证明闭环收敛、优化性能或新的条件包络。

## 真实数据准备与限制

实际调用 `torchvision.datasets.CIFAR10(download=True)` 仍返回 `URLError: Tunnel connection failed: 403 Forbidden`。运行策略自定义允许域为空，而草稿已有 www.cs.toronto.edu/cave.cs.toronto.edu：草稿尚未成为当前运行策略，不能说已恢复访问。没有发现本地 CIFAR 归档或可用批次，因此真实训练未执行，未用合成数据冒充。

见 [本地导入说明](../../docs/CIFAR_IMPORT.md)。`cifar_smoke.py` 可离线导入官方 tar.gz，先验证 torchvision 官方 MD5、安全解压，再做批次校验；也可直接复制官方批次目录。数据就绪后最多训练 32 张真实图像、两个客户端各两个 SGD 步、一次聚合，评估 32 张测试图像。当前 blocked 结果和真实异常保存在 `cifar-smoke.json`/`.log`。

要恢复官方下载，请在环境设置审核保存已有域名草稿并发布，然后重新运行极小检查。无需在聊天中提供 token。未运行 100 轮训练，也没有 FEMNIST 数据或实验结果。

## 测试、文件与复现

本阶段完整测试：**34 项，33 通过，1 项 CUDA 缺硬件跳过**。新增固定/滚动校准公式、区间宽度参数、延迟参考准入、批量与单条 e-process 数值等价、预聚合过滤、全拒绝 no-op、未知漂移反例、错误归档校验等检查。GPU 跳过不能当作已验证。

- 实现：`calibration.py`、`stress_validation.py`、`matched_ablation.py`、`plot_stage3.py`、`build_stage3_report.py`、`cifar_smoke.py`；训练脚本新增可选证据过滤回调。
- 协议：`configs/stage3*.json`，不得使用结果文件调参。
- 原始结果：本目录及 `misspecification/`、`boundary/` 的 `.csv.gz`；每文件 210000 行、每方法/alpha 10000 行。
- 汇总：`all_summary.csv`、各 `results.json` 和 `summary.csv`，含全部区间、漏检、延迟及预测/聚合误差。
- 消融：`matched_prediction.csv.gz`/`.json`、`paired_prediction.csv`。
- 日志：各 `stress.log`、`unit-tests.log`、`matched_ablation.log`、`cifar-smoke.log`。
- 图：`figures/` 中 5 张 PNG 和对应 SVG，分别为误报控制、强度功效、检测延迟、同数据预测消融、候选聚合。
- 理论：`docs/STAGE3_THEORY.md`；来源、配置、原始文件哈希保存在 JSON，运行版本代码在 `source_snapshot/`。

95% 区间均为逐项区间，不是为全部场景/alpha/方法调整的同时区间。没有把经验校准通过、零误报或高检测率替代数学假设证明。
'''
    (base/'STAGE3_REPORT.zh-CN.md').write_text(report)
    print('Saved full report and 399-row combined summary.')


if __name__=='__main__':
    main()
