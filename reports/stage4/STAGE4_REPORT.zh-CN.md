# SETA-FL 第四阶段：理论代码对齐与真实先导门禁报告

日期：2026-11-03。分支 `seta-real-pilot`，基于 `seta-stat-validation` 的 `79c3931`。没有修改 main。

**本阶段完成了可独立完成的代码、理论审查和单元测试。真实数据仍被代理403阻断：官方MD5未实际验证，32张真实图像测试未执行，36个先导组合均未启动。没有真实准确率、误报率或检测率结果。**

## 理论变量与训练代码对齐

完整逐项表和总体误报推导见 [STAGE4_ALIGNMENT.md](../../docs/STAGE4_ALIGNMENT.md)，涵盖第三阶段每个数学变量，包括真实训练中没有已知对应值的真条件均值、锚点证书和滚动参考。

旧 `ClientHistory.predict()` 用历史特征均值定方向，没有上下文 c_t；不能等同于第三阶段的条件仿射预测。主训练入口已切换到 `ContextualHistory`：参考前半部分定方向，后半部分用显式历史上下文拟合线性 ridge 预测。固定 ridge=.001，不用测试标签调参。

新上下文是更新前全局模型参数 RMS 相对初始 RMS 的对数变化，经 tanh 映射到 [-1,1]。轮前构造不可变 `RoundContext`，为所有客户端冻结预测后才进行任何本地训练、攻击、观测和聚合。每个上下文的来源、均值、方向、拟合系数、秩与外推状态均进入客户端日志。

核心对应：X_t=`Prediction.score(features)`；muhat_t=`Prediction.mean`；b_t=`Prediction.mean_error_bound`；a=`monitor.lambdas`；sigma²=`monitor.variance_proxy=1`；log Q=`last_log_factors`；log L=`log_components`；log E=`log_e`。`monitoring_audit.py` 从客户端日志复算指数过程；通过只说明数值/时序一致。

**仍未建立的关系**：新上下文有模型反馈，不是第三阶段的已知外生路径；真实更新没有独立±1锚点、稳定仿射条件均值或1024个独立参考。前后半拆分不消除轮间依赖。默认 b=.2 仍是研究假设，预留 delta 也不能证明校准失败概率。真实条件漂移适应能力没有得到验证，不作此主张。保守 b=1+abs(muhat) 虽确定性覆盖，但无有用检测力。

## 客户端状态与三种未校准策略

状态为 `UNCALIBRATED`、`MONITORED`、`ALARMED`；没有 NORMAL。`MONITORED` 只表示参考/预测可用、检验已启动，并不表示统计校准得到证明；日志另存 `ASSUMED_NOT_CERTIFIED` 等状态。ALARMED 锁存且权重为0。

以下是十个等样本量、全部未校准客户端的**确定性代码契约比较**，不是 CIFAR-10 实验：

| 策略 | 每客户端权重 | 客户端总量 | 服务器保留量 | 状态 |
| --- | --- | --- | --- | --- |
| 暂缓 defer | 0 | 0 | 1 | 全部 UNCALIBRATED |
| 受限 capped | .02 | .2 | .8 | 全部 UNCALIBRATED |
| 声明正常预热 clean-warmup | .1 | 1 | 0 | 预热阶段仍全部 UNCALIBRATED |

预热必须显式声明，固定前8轮禁用攻击。每次观察先进入隔离区；普通 observe 不会添加正常参考。只有单独的、限定在声明预热期的调用可添加协议参考，且来源标为 `DECLARED_PROTOCOL_ONLY`。第9轮才可能转为 MONITORED，若参与不足、没有足够记录则继续未校准。监控开始后，未报警或可疑更新均不能自动提升为正常。

受限权重保留服务器质量，避免 `apply_aggregate` 再归一化使限额失效。暂缓策略在无外部参考时可能一直不进行全局学习，这是必须报告的冷启动困境。正常预热在真实部署中若可能受攻击，不能被当成事实。权重限额不等于约束任意大恶意更新的影响。

每轮记录所有客户端（含缺席者）的前后状态、选中标志、实际权重、服务器保留量、上下文、参考数量/来源、muhat/b/X/E、过程次数及各自错误预算。当前只有单元/契约记录，没有伪造真实每轮记录。

## 总体错误预算

令 N 为固定客户端数，K 为分别报警的特征检验数，J 为预定实验组合数，rho=.1：

alpha_e=(1-rho)*alpha_global/(N*K*J)，delta=rho*alpha_global/(N*K*J)。

每项在包络以失败概率≤delta成立且满足可预测性/有界条件时，Ville 加校准失败事件给出至少一次误报概率≤alpha_e+delta；对所有客户端、特征、组合取并集，上界为 alpha_global，不要求检验彼此独立。

先导固定 alpha_global=.05、N=10、K=1、J=36：每检验 alpha_e=.000125、delta≈.000013888889、报警阈值8000；全家族序贯预算 .045、校准预算 .005，总和 .05。默认单次10客户端则为 alpha_e=.0045、delta=.0005。参与率变化不会回收并放大预算。

16维特征投影成一个标量，所以 K=1；十个固定下注的有效混合也是一个过程。若增加多个独立报警特征，则必须分K份或证明合并过程；当前控制器遇到 K!=1 直接拒绝，不能静默复用预算。重复实验已按36组合分配；已启动的实验家族禁止无状态重跑，避免证据重置和重复消费预算。

**预算算术已验证，真实概率保证仍未成立。** 当前 assumed 模式没有证明 P(包络失败)≤delta；不能凭分配数字就声称总体误报率≤.05。该概率结论不包括未校准准入损害、漏检或预热期真的正常这一未经证明的前提。

## 官方数据访问与完整性

本阶段实际重试官方 HTTPS HEAD，并通过 `torchvision.datasets.CIFAR10(download=True)` 重试真实加载，两者均在代理 CONNECT 阶段收到403。运行策略当前自定义允许域为空，配置草稿则已有 `www.cs.toronto.edu`、`cave.cs.toronto.edu`：草稿没有在当前运行策略生效。

未发现本地官方归档或可用批次。官方归档预期MD5为 `c58f30108f718f92721af3b95e74349a`；这是预期值，**不是实际文件已验证**。`cifar-smoke.json` 明确记录 archive_md5_verified=false、batch_md5_verified=false、smoke_passed=false，以及真实URLError。没有使用合成数据替代。

本地导入方式详见 [CIFAR_IMPORT.md](../../docs/CIFAR_IMPORT.md)：

```bash
source .venv/bin/activate
python cifar_smoke.py --import-archive /workspace/library-files/cifar-10-python.tar.gz --root ./data_official --out reports/stage4/cifar-smoke.json
python pilot_cifar.py --root ./data_official
```

导入器先比对官方归档MD5，安全解压且拒绝覆盖已有文件；也支持复制官方完整批次目录后离线使用。若只有批次没有归档，会分别验证每个官方批次/元数据MD5并明确说明没有归档可验证，不会伪称归档校验通过。

若选择恢复在线访问，需要在环境设置审核、保存并发布已有域名草稿，然后再次运行官方门禁；当前工具保存草稿不等于应用运行策略。没有索要GitHub token或绕过代理。

## 指定32图像真实测试

`cifar_smoke.py` 已实现以下可检查步骤，但**由于数据门禁失败，本次全部未运行**：

1. 验证50000/10000条官方训练/测试记录、图像3×32×32、标签范围及文件完整性。
2. 取前32张真实训练图像，两个客户端各16张，batch=8，各一轮；用实际 optimizer 计数断言各2步，非硬编码假称执行。
3. 检查数据/标签/模型设备一致、更新有限非零、本地训练不修改服务器。
4. 聚合一次，检查权重 .5/.5、每个参数与独立闭式一致、服务器设备与参数有限。
5. 只在32张真实测试图像上作功能评估，保存实际步数、损失、设备、权重、聚合误差和完整性指纹，不将其当作准确率基准。

## 先导协议与实际状态

协议先于真实结果提交于 `ae14c8d`：CIFAR-10、SmallCNN、10客户端、20轮、3种子；IID/Dirichlet(.5)，正常/Sign Flipping，三种未校准策略，共36个组合。为控制CPU先导规模，每次仅随机选2000张真实训练图像、固定前1000张真实测试图像。不能将该子集实验当作全量CIFAR基准。

正常预热策略前8轮声明禁用攻击，第9轮才开始攻击。其他策略不从目标生成参考，因此可能一直未校准，检测率必须与监控覆盖率一起报告。分类测试标签只用于最终准确率，不用于预测器、界或阈值选择；恶意ID/攻击时点仅用于实验控制及事后统计。

状态文件：`BLOCKED_AT_REAL_DATA_GATE`，runs_started=0、runs_completed=0、metrics_available=false。`run_registry.csv` 的36行均为 NOT_RUN；`planned_commands.json` 只是将来由通过门禁的运行器调用的命令，不是已执行日志。

因此正常客户端误报率、恶意检测率、全局准确率和真实训练稳定性都为**未测量**，没有写入0、伪造CSV或生成虚假曲线。将来统计会分别给出所有客户端被标记比例、已实际监控客户端的误报/检测率及覆盖率；没有任何监控对象时为不适用，不假称误报率0。

## 哪些条件得到验证

| 条件或能力 | 本阶段证据 | 真实数据结论 |
| --- | --- | --- |
| 参数/上下文先于当前更新冻结 | 单元测试与主循环调用顺序 | 代码实现已核对；真实路径尚未执行 |
| X的范围、sigma²=1 | 特征范数、单位方向、代码断言及测试 | 结构性有界依据存在；没有真实分数样本 |
| e-process递推、报警锁存 | 日志复算与篡改失败测试 | 数值实现已验证；不证明条件期望 |
| 状态不等于正常、隔离不自动入库 | 三状态、预热限制、重复观察拒绝测试 | 代码契约已验证，真实策略收益未比较 |
| 限额与服务器保留质量 | 有限权重/真实参数聚合单元测试 | 代码算术正确；未证明鲁棒聚合或收敛 |
| 全体错误预算之和≤.05 | 客户端×特征×组合预算测试 | 算术正确，真实校准失败界未证明 |
| 稳定仿射真条件均值、独立参考 | 不存在真实验证证据 | 仍为研究假设 |
| b=.2覆盖真实条件预测误差 | 未获得证书或真实数据 | 仍为研究假设，不能声称漂移适应 |
| 官方MD5、指定真实小测试 | 下载403，完整性标志均为false | 未验证、未运行 |
| 三种子真实先导性能 | 36组合全未启动 | 无实验结果 |

## 测试与交付文件

完整单元测试 **50项：49通过、1项因无CUDA跳过、0失败**。单元测试使用明确的张量/临时文件夹fixture，不是CIFAR结果；日志中的fixture损失和假数据门禁错误不能解释为真实实验。

新增 `client_control.py`、`monitoring_audit.py`、`pilot_cifar.py`、`compare_client_policies.py` 及相应测试；修改真实训练入口、e-process显式支持参数、实际SGD步数计数和完整性门禁。源码哈希/版本/测试状态见 `validation_metadata.json`。

- 理论映射：[STAGE4_ALIGNMENT.md](../../docs/STAGE4_ALIGNMENT.md)。
- 协议：`configs/stage4_pilot.json`、`frozen_plan.json`、`family_budget.json`。
- 真实失败证据：`official-download.log`、`cifar-smoke.log/.json`、`pilot.log`、`pilot_status.json`。
- 状态契约比较：`policy_contracts.json`、`policy-contracts.log`；明确没有数据实验。
- 已准备但未执行：`run_registry.csv`、`planned_commands.json`。
- 测试：`unit-tests.log`；复现 `python -m unittest discover -s tests -v`。

下一步需要先恢复官方域名运行策略或导入官方数据，然后重新运行 `python pilot_cifar.py --root ./data_official`。门禁通过后运行器才会自动启动已声明的36组合。当前不批准任何真实漂移适应或严格误报控制主张，也没有运行100轮实验。
