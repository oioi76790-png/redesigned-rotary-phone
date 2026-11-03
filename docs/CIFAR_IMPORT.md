# CIFAR-10 本地导入与极小训练验证

当前云环境运行策略没有允许 `www.cs.toronto.edu`；官方 HTTPS 请求在代理 CONNECT 阶段返回 403。这是网络目的地主机策略拒绝，不是 CIFAR 数据损坏或 GitHub token 缺失。之前草稿中的域名不等于已发布的运行策略。

方案一：在环境设置审核保存并发布 `www.cs.toronto.edu` 与下载重定向 `cave.cs.toronto.edu`，然后运行：

```bash
source .venv/bin/activate
python cifar_smoke.py --download --root ./data
```

方案二：在可访问官方下载地址的本地机器下载官方 `cifar-10-python.tar.gz`，将原始文件复制/上传至云工作区，再执行：

```bash
python cifar_smoke.py --import-archive /workspace/library-files/cifar-10-python.tar.gz --root ./data
```

程序先比对 torchvision 中的官方归档 MD5，再使用限制路径的安全解压；不匹配就退出，不接受自行伪造的同名文件。目标目录中若已有个人修改文件，请选新的 `--root`，避免官方归档覆盖它们。

方案三：直接复制官方解压后的完整目录，布局如下：

```text
data/cifar-10-batches-py/
  data_batch_1 ... data_batch_5
  test_batch
  batches.meta
```

然后离线检查，不进行任何网络请求：

```bash
python cifar_smoke.py --root ./data --check-only
python cifar_smoke.py --root ./data
```

torchvision 对批次文件执行内置完整性校验。最后一个命令仅用真实训练集前 32 张图片，两个客户端各 16 张、各两个 SGD 步、一次全局聚合，并评估 32 张真实测试图片。它不是完整训练，不验证准确率、攻击检测率或条件假设。不要为得到漂亮的统计值挑选子集或调整算法。

结果现在默认写入 `reports/stage4/cifar-smoke.json`（第三阶段归档保留）；如果数据不可用，会记录 blocked 和真实异常并以退出码 2 结束，不退回合成数据冒充真实结果。若导入目录已有文件，导入器拒绝覆盖，请选新的 `--root`。

第四阶段增强了实际检查：分别记录归档 MD5 与每个官方批次/元数据文件 MD5；若只导入批次目录，会明确标记“没有归档可校验”，不会伪称归档校验通过。训练用实际 optimizer 步数计数验证每客户端恰好两步，核对模型/数据/标签设备、非零有限更新、本地训练不修改服务器、权重 .5/.5、逐参数聚合闭式以及有限评估结果。

真实先导实验必须通过该门禁后才能启动：

```bash
python pilot_cifar.py --root ./data --download
# 已本地导入时不需要网络：
python pilot_cifar.py --root ./data
```

该入口计划10客户端、20轮、三个种子、IID/Dirichlet、正常/Sign Flipping及三种未校准策略，共36个组合。它先重新执行真实数据门禁，失败时所有组合保持 NOT_RUN。默认只使用2000张真实训练图像和1000张真实测试图像，参数预先固定在 `configs/stage4_pilot.json`，不使用测试标签调参。

已启动的实验家族禁止无状态重跑，避免重置证据和重复消费同一错误预算；中断恢复尚未实现。仅数据门禁失败、没有启动任何组合的任务，可在恢复数据后继续调用同一入口。
