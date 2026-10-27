# 共享GPU dispatch契约

此内部契约由[GPU编排](<gpuTransformer.cpp>)使用，所有buffer为连续FP32，句柄为executor所属的size_t，不进入公开API。Vulkan实现采用[shader目录](<../vulkan/shaders/prepareQkv.comp>)的GLSL布局，各API独立翻译绑定与push机制，不改变下列顺序。元信息以FP32整数保存，groupIds由编排压缩为连续组号，不直接转换外部大整数。

## 生命周期

模型构造begin后upload(values,true)全部权重和单元素零buffer，end完成上传并保留persistent。调用begin后upload(values)和allocate(count)均为scratch。dispatch只记录设备操作且必须保证先前写入对本次读取可见，不能隐式回读。download提交此前命令并等待，返回该buffer全部FP32，但保留scratch和句柄供继续dispatch。end提交剩余命令、等待、释放scratch，persistent保持有效。GpuTransformer用recursive_mutex保护整个begin/end区间。

## Dispatch

下表是十个v1核的共享契约，不代表Vulkan完整包仅有十核。除gemm外local_size_x=64、y=z=1；行核与attentionFused按行/query分派，其余按输出元素数。支持二维展开的元素核index=globalID.y*numGroups.x*localSize.x+globalID.x，行核row=workGroupID.y*numGroups.x+workGroupID.x。旧silu验证核只看X，不直接用于超过65535组的二维grid。binding从0开始，push每项4字节，无隐含padding，float按位放入uint32容器。

| kernelName | bindings | push顺序 | groups |
|---|---|---|---|
| gemm | input, weight, bias, output | rows, columns, inner, useBias | ceil(columns/16),ceil(rows/16),1 |
| rmsNorm | input, scale, output | rows, width, epsilon(float) | rows,1,1 |
| softmax | input, output | rows,width | rows,1,1 |
| silu | input, output | count | ceil(count/64),1,1 |
| prepareQkv | fused, rotary, q,k,v | tokens,heads,headDim,ropeDim,isTime | ceil(tokens*heads*headDim/64),1,1 |
| mergeKv | prefix,tail,output | rows,count,base,dim | ceil(rows*(base+count)*dim/64),1,1 |
| elementwise | first,second,output | count,width,mode | ceil(count/64),1,1 |
| attentionScores | q,keys,metadata,scores | rows,count,base,encoded,contextPatches,heads,headDim,covariates,isTime | ceil(rows*count*heads*keyCount/64),1,1 |
| attentionValues | probabilities,values,output | 同attentionScores | ceil(rows*count*heads*headDim/64),1,1 |
| attentionFused | q,keys,values,metadata,output | 同attentionScores | rows*count*heads,1,1 |

## Vulkan专用扩展

当前Vulkan包为26个计算核及各自Budget变体，共52模块，完整绑定、localSize、push字段与限制见[manifest](<../vulkan/shaderManifest.json>)。有序精度路径、共享前缀和组内注意力按executor能力选择，其他GPU API保留原核。G1投影加速保留，原在线归约核供独立诊断或超范围回退，不把候选当作跨API必需核。

siluGrid与silu绑定、push相同，但正确展平二维grid。GpuTransformer的可选residentResidual在Vulkan工厂也默认关闭，显式实验M≥128使用siluGrid，其它API构造不启用。GpuExecutor的生命周期和buffer布局没有改变。

预算仅Vulkan执行器实现，0保留原派发、非0为256倍数/u32内。固定local_size不改，NumWorkgroups仅作展平步距的核规范化到固定W，Budget常量255用于该步距；native核保留坐标。末行严格保留覆盖范围，窗口间执行依赖，完整行在预算内合并，每operator一个管线，不按输入长度无限特化。只约束本模型compute逻辑窗口，不保证物理EU或带宽配额。

## 布局与数值

- gemm的weight为(columns,inner)，输出input乘weight转置。useBias=0不访问bias，允许绑定单元素零buffer。点积按K顺序FP32逐项乘加，不收缩FMA。
- rmsNorm维持(input*inverse)*scale顺序，softmax允许负无穷且全屏蔽行输出0。参考阈值不变。
- prepareQkv输入(token,3,headDim,heads)，输出(token,heads,headDim)。rotary为(tokens,ropeDim/2,2)，末维依次cos、sin，CPU按reference的FP32角度及libm生成。仅isTime=1旋转Q/K的j和j+headDim/2，V不旋转。
- mergeKv将(rows,base,dim)和(rows,count,dim)拼成(rows,base+count,dim)，只读前缀。Vulkan可用范围与别名检查后的transfer copy替代计算拼接。shared路径还可直接读取一份(prefixRows,base,dim)前缀及(rows,count,dim)尾部，不在CPU或GPU物化rows份历史。
- elementwise mode=0输出first+second，mode=1输出first*sigmoid(second)，mode=2从first(row,2*width)输出前半*SiLU(后半)，count为输出元素数且second不读取。
- attention的q/output均(rows,count,heads*headDim)。isTime=1时keys/values为(rows,base+count,heads*headDim)，keyCount=base+count。isTime=0时keys/values为(rows,count,heads*headDim)，keyCount=rows，沿同一local slot跨row做注意力。
- attentionFused每query/head一个64线程workgroup，按64键tile在线计算最大值、分母和PV，工作区只有128个共享FP32，不分配T平方scores。共享编排使用此核，分解Scores/Values保留为数值诊断核。
- scores/probabilities为(rows,count,heads,keyCount)。metadata前三段为keyValid(rows,encoded)、roles(rows)、compressedGroupIds(rows)，原核只读这三段。组内及有序核继续读取groupStart(rows)、groupWidth(rows)，旧偏移不变。roles为Target=0、PastCov=1、KnownCov=2，encoded=0时首段为空。交错groupIds回退全行范围并按原掩码过滤，不把同组假设扩展为任意输入。
- prepareQkv、mergeKv、attentionScores、attentionValues与elementwise的binding/push精确字段声明以对应GLSL为准，Vulkan不依赖开发机绝对shader路径。
