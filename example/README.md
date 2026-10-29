# 示例说明

每个示例是独立CMake工程，根工程不再构建示例。运行目录为仓库根目录，模型目录用相对路径。

~~~powershell
cmake -S example/<目录> -B example/<目录>/build
cmake --build example/<目录>/build --config Release
~~~

## timestralExample（example/main）

读取本地模型，生成128步正弦输入，用默认后处理预测32步，将全部分位数写成CSV到标准输出，将加载及预测耗时写到标准错误。它不读取真实业务序列，不包含模型下载逻辑。

~~~powershell
.\example\main\build\Release\timestralExample.exe model/timestral-5m 32
~~~

## timestralStream（example/stream）

模拟100Hz（10ms）传感器：600点正弦加确定性噪声，每30点以固定streamId调用submit并等待future，策略为Mailbox。每次输出样本数、复用页数、新算页数、命中率、prefill/decode/total耗时和首个预测中位数；结束时给出平均单次耗时、平均命中率，以及同数据用Timestral::predict逐点全量重算的总耗时与加速比。

~~~powershell
.\example\stream\build\Release\timestralStream.exe model/timestral-5m
~~~

模型至少需要2×patch_size个样本（5M为64）才能预测，因此前60点只入流不预测，表格首行是90点。输出为纯文本、无ANSI、退出码0。

## timestralSignal（example/signal）

用噪声加多个正弦的傅里叶叠加信号验证真实5M权重能否延续周期结构：以context长度的带噪观测为上下文，预测horizon步，与未来horizon步的无噪声确定性真值对照。它只依赖标准库，不操作终端模式、不读键盘、不发ANSI转义，输出可安全重定向到文件或管道。

~~~powershell
.\example\signal\build\Release\timestralSignal.exe model/timestral-5m
~~~

命令行参数：--seed N、--context N、--horizon N、--f0 F、--harmonics K、--noise S；默认f0=0.0078125（周期128步=4个patch）、harmonics=3、noise=0.1、context=256、horizon=128、seed=42。超出允许范围的参数钳制到边界：f0取[1/2048,1/64]，harmonics取[1,6]，noise取[0,0.5]，context取[64,模型contextLength]，horizon取[32,512]。

输出依次为参数行、指标行、判定行、固定100列宽的ASCII曲线和符号图例，全部为纯文本。一次运行结束即退出：成功返回0，参数错误返回2，模型加载或预测失败返回1。

指标含义：

- MAE、RMSE：预测中位数与无噪声真值之间的平均绝对误差与均方根误差，量纲同读数
- corr：两者逐点相关系数，衡量周期相位是否对齐
- ampRatio：预测中位数标准差与真值标准差之比，接近1表示幅值没有明显衰减或放大
- cov80、cov90：真值落在模型80%与90%中心区间内的比例，接近0.8与0.9说明区间校准合理
- predict_ms：本次predict调用的耗时

该模型的patch_size=32：每条通道32个数值槽位及其缺测信息投影为一个时间token，输出头每个token仍生成32个逐点预测及分位数。不能把patch大小当作原始采样间隔，也不能直接以2个patch作为严格的Nyquist界限；此示例选择较长周期只是实验设置。

## timestralTui（example/tui）

[界面](<tui/streamingTui.cpp>)与[实验实现](<tui/streaming.cpp>)从零历史开始，使用[JSON配置](<tui/streaming.json>)生成可重复采样的多通道信号。

~~~powershell
cmake -S example/tui -B example/tui/build
cmake --build example/tui/build --config Release --parallel
.\example\tui\build\Release\timestralTui.exe model/timestral-5m example/tui/streaming.json --threads 4
~~~

在真实终端中，直接按数字1至9立即推进对应通道，不需要Enter；a推进一组、r重算、b切换p10/p90带、q或Ctrl+C退出。--threads N设置所选BLAS、Eigen或ytensor后端的计算线程数，省略时用1线程（串行），0为自动值（最多4），底部状态行显示实际线程数。本机512窗口/horizon512下15次更新约：1线程6.63秒、4线程4.14秒、16线程3.88秒；单次N约364/219/249ms、K约160/110/119ms，剩余成本主要是N的原生全窗重算，不是缓存解码。16线程只在TUI这类较大批量上略快，在小GEMM回归上会明显过慢，因此自动值封顶4，需要时可显式指定。自动启用彩色替代屏幕与原地刷新，退出恢复终端输入模式和光标；进度条C绿/A黄/P红，原生预测青色、FIFO预测品红，分位带为相应暗色。窗口尺寸变化时调整图形大小，颜色转义不占列宽。平台键盘与终端接口只在[示例终端模块](<tui/terminal.cpp>)中，不进入推理库。实验期的timestralRopeTui与它完全相同，发布前已删除，只保留timestralTui。

输出重定向、输入管道或--plain会自动使用无控制符纯文本模式，便于脚本；--once只打印初始画面，--no-bands启动时隐藏区间。正常交互不必改JSON的color开关。Windows即时按键和恢复已经实测，POSIX终端适配已提供但本机没有Linux/macOS运行验证。

配置包含window、horizon、advance、interval、width、height、seed、color和channels。window是cached最长保留的原始采样槽数，horizon=aux+predict，advance是每次按键推进的时间槽数；三个长度都须为模型patch倍数，window至少2patch，window+horizon不得超过模型上下文预算。8k窗口可设置window=8192，这不是8192个模型token，真实5M只有256个patch。通道推进差不得大于horizon，超过时先推进短通道。

每个通道配置terms=[{a,b,c,d},...]，其值为多个a*sin(b*t+c*randn+d)之和，再加noise*randn+bias。b单位为弧度/秒，c为相位噪声，通道noise为幅值噪声。sampleHz控制真实采样频率，sampleJitter控制采样间隔的对数正态扰动，保证时间递增。样本落入公共网格桶，同桶保留最后一个，不插值也不删除空列。匹配通道中任意若干terms的a/b/d即可手工构造共享分量，不要求共享的是主谐波。无噪声reference将两类randn系数置零，但保留bias。

只绘制第一通道：R为真值、N为每轮原生全量条件重算p50、K为冻结KV+FIFO实验p50，*表示曲线重合。cached区间不再重复画N、K两条相同的输入曲线，而用o表示二者共享的带噪输入；o与R分行时是观测噪声相对无噪声真值的偏差，不是把预测写进了KV。同时绘制两路真实分位数头的p10至p90带，并显示首个输出位置的p10/p50/p90数值，不使用RMSE或高斯假设伪造区间。已知位置区间退化为原始读数，不解释为概率估计。进度轴及绘图区只使用ASCII单列字符，C/A/P分别是cached/aux/predict，所有行共用同一物理时间轴。两条预测的RMSE只统计未观测位置，不把原始已知读数作为模型预测精度。零历史及共同历史不足2patch时明确标注insufficient history，不生成伪预测。

N调用既有Engine原生分段条件路径，K只对新共同历史patch编码入持久页；aux及生成值不进入持久KV。K重算条件尾部时读取冻结前缀，FIFO只删除最老页、不重算剩余页，报告retained/reused/appended/fifo_evicted和包含追加页成本的耗时。为隔离缓存差异，两条路径都使用1patch反馈、关闭对称平均和非负截断；K稀疏aux采用patch级近似，N保持逐位置观测分段。

这条近似路径的质量与时延结论、未滑动控制线与现象分析见[推理侧设计](<../doc/zh/inference.md>)的冻结前缀与FIFO滑窗实验一节。
