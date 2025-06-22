本篇用于学习使用devito完成FWI的方法
## 模型 setup
$$\begin{align*}
\mu\frac{\partial H_x}{\partial t} &= -\frac{\partial E_z}{\partial y} \\
\mu\frac{\partial H_y}{\partial t} &= \frac{\partial E_z}{\partial x} \\
\varepsilon\frac{\partial E_z}{\partial t} + \sigma E_z &= \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} + J_z
\end{align*}$$

## PML层设置
![[Pasted image 20250621102144.png]]
![[Pasted image 20250621102235.png]]
