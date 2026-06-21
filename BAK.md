• CD-PiF 相比原始 PiF 的创新点主要在 4 个层面。

  1. 单源 PI → 多源共识 PI

  原始 PiF 用一个源模型估计 perceived-importance，通常是 BERT 或单个 CLM。CD-PiF 用多个现代指令模型共同估计：

  Llama-3.1 + Mistral + Qwen

  并同时计算：

  mean PI：平均重要度
  var PI：跨源不稳定性

  候选位置不再只是“PI 低”，而是：

  PI 低 + 方差低

  也就是优先改那些多个模型都认为“不关键”的稳定位置。这对应“迁移性来自跨模型共享表征，而不是源模型特异性扰动”。

  2. 单模板 intent → 多模板 intent

  原始 PiF 主要依赖类似：

  This intent is [MASK]

  这种单一 intent 模板。CD-PiF 改成多模板：

  The user intent is
  This request asks for
  The safety category is

  这样 PI 不再过拟合某一个模板措辞，方法上更科学，也方便做模板消融。

  3. 单一 intent 信号 → harmfulness/refusal 解耦代理

  原始 PiF 把“模型如何理解意图”混成一个信号。CD-PiF 引入三组 label logits：

  harm_labels: harmful, unsafe, malicious
  refusal_labels: refuse, decline, cannot
  benign_labels: benign, safe, allowed

  从而近似分离两个潜变量：

  harmfulness：模型是否识别有害性
  refusal：模型是否倾向拒答

  这比单纯 flatten intent 更有解释性，因为越狱迁移里“识别有害”和“触发拒答”并不是同一个机制。

  4. token 替换 → span 级编辑 + 语义门

  原始 PiF 基本是单 token 替换。CD-PiF 支持 span 级候选，最长可设：

  --span_max_len 3

  并加入语义保持门：

  hidden-state token matching similarity
  slot preservation

  所以它允许更自然的局部短语改写，同时避免语义漂移。

  一句话总结：

  PiF = 单源、单模板、单 token、单 intent 信号的 PI flatten
  CD-PiF = 多源共识、多模板稳健估计、harmfulness/refusal 解耦、span 级语义受控编辑

  论文写法上，可以把 PiF 看成 CD-PiF 的特例：

  source=1, template=1, span_len=1, latent_beta=0

  这样 CD-PiF 不是简单 “PiF++”，而是一个更一般的 transfer-stable importance editing 框架。

