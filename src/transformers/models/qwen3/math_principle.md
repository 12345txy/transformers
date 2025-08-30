* 原始实线性：$Y = XA$，其中
  $X \in \mathbb{R}^{N \times (2m)}$, $A \in \mathbb{R}^{(2m) \times (2n)}$, $Y \in \mathbb{R}^{N \times (2n)}$。
* 配对为复数：把最后一维配成实/虚两半
  $x = x_{re} + ix_{im} \in \mathbb{C}^m$, $y \in \mathbb{C}^n$。
* 目标：存在唯一 $U, W \in \mathbb{C}^{n \times m}$ (实部/虚部分别为 $U_{re}, U_{im}, W_{re}, W_{im}$)，使得

  $$
  = Ux + W\bar{x} \iff \begin{bmatrix} \Re y \\ \Im y \end{bmatrix} = (B_1(W) + B_2(U)) \begin{bmatrix} \Re x \\ \Im x \end{bmatrix}
  $$

  且该实块矩阵恰等于原来的A。

其中

$$
B_1(W) = \begin{bmatrix} W_{re} & W_{im} \\ W_{im} & -W_{re} \end{bmatrix}, \quad B_2(U) = \begin{bmatrix} U_{re} & -U_{im} \\ U_{im} & U_{re} \end{bmatrix}
$$

分解/还原公式 (唯一)：把A分 $2 \times 2$ 块 (每块 $n \times m$)：

$$
A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}, \quad U_{re} = \frac{1}{2}(A_{11} + A_{22}), \quad U_{im} = \frac{1}{2}(A_{21} - A_{12}),
$$

$$
W_{re} = \frac{1}{2}(A_{11} - A_{22}), \quad W_{im} = \frac{1}{2}(A_{12} + A_{21})
$$

---
