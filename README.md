# triton 学习笔记

跟着官方 tutorial 一路啃下来的练习代码，从 06 号开始是自己加的。
环境：WSL2 + triton 3.8.0，每个文件直接 `python xx.py` 就能跑（要 GPU），跑完会自动和 torch 对照数值。

## 目前写了啥

- vector_add / fused_softmax / replicate.py —— 最开始跟 tutorial 01/02 敲的，softmax 那个的 benchmark 结果在 softmax-performance.png
- 06 matmul：分块 + tl.dot，加了 autotune 和 L2 swizzle。4096³ 能跑出 cuBLAS 一半左右的性能
- 07 layernorm：第一次手写 backward，dW/dB 用 atomic_add 累
- 08 flash attention：就前向。核心是 online softmax，每扫一块 K 都要把旧的 acc 按 exp(m_old-m_new) 缩放一遍
- 09 转置 + persistent kernel：转置比 torch 的 x.T.contiguous() 快 3 倍（259 GB/s vs 84），主要是 coalescing 的功劳
- 10 前缀和 + 直方图：associative_scan 和两遍扫描；直方图做了个对比实验，8 个 bin 的时候朴素 atomic 要 1325us，先在寄存器里累计再合并只要 66us，20 倍
- 11 TMA 版 matmul：用 TensorDescriptor 替掉手工指针，K 的地址循环都不用自己写了
- 12 融合交叉熵：流式 logsumexp，不用物化 (N, V) 的中间矩阵，8192x32K vocab 下比 eager 快 3.1 倍
- 13 add+layernorm 融合，14 silu*mul（swiglu 的激活部分，反向自己推的）
- 15 RoPE、16 int8 量化、17 MoE routing
- 18 fused adamw：和 torch.optim.AdamW 对了 5 步，误差 2e-6
- 19 split-K GEMM：瘦矩阵（16x4096x16384）上比 cuBLAS 还快一点
- 20 causal flash attention：把 K 循环拆成对角块左边/对角块两段，右边根本不用算
- 21 flash-decoding：decode（q 长度=1）时按 KV 的 seq 维切 split，两阶段归并，32K cache 下比 SDPA 快一倍
- 22 rmsnorm：LLaMA 用的那种，反向自己推的，和 torch.nn.functional.rms_norm 对过
- 23 把 kernel 注册成 torch custom op（torch.library.custom_op + register_fake），能过 torch.compile 的 trace

## 踩过的坑（都留在代码注释里了）

1. `tl.arange` 的参数必须是 constexpr，不能存到普通变量再传进去（15 号卡过）
2. 20 号那个最坑：块指针 `tl.advance` 推进的是函数内的局部变量，把 K 循环拆成两段之后，第二段拿到的还是原始指针，等于对角块把前面的列重复算了一遍。误差只有 0.1 左右，一开始以为是精度问题，查了半天发现是重复计数。教训是 jit 函数里跨"逻辑段"共享的指针要么传出去要么别拆段
3. autotune 要装饰在 @triton.jit 的 kernel 上，套在普通 python 函数上会报 arg_names 错误
4. layernorm 反向第一版忘了乘 w，dX 全错；另外直方图那种"逐 bin 循环标量比较"的写法结果会全错，要写成广播比较
5. 量化的误差别用相对误差看，x 接近 0 的元素相对误差必然爆炸，应该对照 absmax/254 这个理论界
6. 21 号卡我最久：一是 q 和 KV 的平面 stride 不一样（q 是 D，KV 是 N*D）不能共用；二是被 mask 的 k 行算出来 qk=0 不是 -inf，会把 softmax 分母污染；三是归并公式 acc 不能乘 l。三个 bug 叠在一起，误差 1e1 量级
7. silu 的导数我背错了：dsilu = sig*(1 + x*(1-sig))，sigmoid 的导数是 sig*(1-sig)，不是 sig*sig

## 没写的

flash attention 的反向（tutorial 06 后半段），还有 split-K 的 stream-K 版本，以后有空补。
make_block_ptr 在 3.8 里已经标弃用了，新的写法是 tl.make_tensor_descriptor，11 号里有示例。
