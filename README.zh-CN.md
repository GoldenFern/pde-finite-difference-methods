# 微分方程数值解课程大作业

[English](README.md) | **简体中文**

上海交通大学 2023-2024 学年春季学期  
微分方程数值解（MATH3808-2）课程大作业

本仓库收录两次本科课程大作业的报告、代码和数值结果。两个项目分别讨论二维 Poisson 方程的多种数值解法，以及对流方程中几种差分格式的稳定性和间断点表现。

## 项目一：五点差分法求解 Poisson 方程中不同数值解法的比较分析

求解矩形区域上带齐次 Dirichlet 边界条件的二维 Poisson 方程。测试算例为

$$
\Delta u = -2\pi^2\sin(\pi x)\sin(\pi y),\qquad u=0\ \text{on}\ \partial\Omega,
$$

其解析解为 $u=\sin(\pi x)\sin(\pi y)$。

报告比较了以下数值方法：

- 五点差分格式
- 离散正弦变换（DST）
- LU 分解
- Jacobi 迭代（双重循环实现）
- Jacobi 迭代（`scipy.signal.convolve2d` 卷积加速）
- Gauss-Seidel 迭代
- SOR 迭代

实验统计了不同网格划分下各方法的运行时间、平均绝对误差和最大误差，并进一步把方法推广到非齐次 Dirichlet 边界条件。

主要结论：DST 的运行速度明显快于其他方法；DST 与 LU 分解得到相同的离散解，误差随网格加密按 $O(h^2)$ 下降；迭代法中卷积加速的 Jacobi 方法明显快于普通循环实现。

文件：

- [报告 PDF](Project1/Project1.pdf)
- [报告 Markdown](Project1/Project1.md)
- [报告 DOCX](Project1/Project1.docx)
- [Python 代码](<Project1/微分方程数值解 - Project1.py>)

## 项目二：差分格式稳定性及数值效应比较实验

求解一维对流方程

$$
u_t+au_x=0,
$$

初值为单位阶跃函数，空间步长 $h=0.1$，时间步长 $\tau=0.08$，网格比 $\lambda=\tau/h=0.8$，计算到 $t=4.0$，速度分别取 $a=1,2,4$。

报告实现并比较了四种差分格式：

- 迎风格式（Upwind Scheme）
- Beam-Warming 格式
- Lax-Friedrichs 格式
- Lax-Wendroff 格式

实验计算了离散 $L^2$ 误差，结合增长因子和修正方程分析了各格式的收敛条件，以及间断点附近的耗散和色散现象。

主要结论：$a=1$ 时四种格式都收敛；$a=2$ 时只有 Beam-Warming 格式收敛；$a=4$ 时四种格式都发散。迎风格式和 Lax-Friedrichs 格式以耗散为主，Beam-Warming 格式和 Lax-Wendroff 格式在间断点附近表现出明显的色散振荡。

文件：

- [报告 PDF](<Project2/Project2 差分格式的稳定性分析.pdf>)
- [报告 Markdown](<Project2/Project2 差分格式的稳定性分析.md>)
- [报告 DOCX](<Project2/Project2 差分格式的稳定性分析.docx>)
- [Python 代码](<Project2/差分格式求解代码.py>)
- `Project2/数值结果/`：四种格式在 $a=1,2,4$ 下的结果图

## 目录结构

```text
NPDE/
├── README.md                 # 英文说明（默认展示）
├── README.zh-CN.md           # 中文说明
├── LICENSE
├── requirements.txt
├── .gitignore
├── Project1/
│   ├── 微分方程数值解 - Project1.py
│   ├── Project1.docx
│   ├── Project1.pdf
│   ├── Project1.md
│   └── media/                # Markdown 报告使用的图片
└── Project2/
    ├── 差分格式求解代码.py
    ├── Project2 差分格式的稳定性分析.docx
    ├── Project2 差分格式的稳定性分析.pdf
    ├── Project2 差分格式的稳定性分析.md
    ├── media/                # Markdown 报告使用的图片
    └── 数值结果/             # 程序生成的 12 张结果图
```

## 运行环境与运行方式

需要 Python 3.9 或更高版本。安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行项目一：

```bash
python "Project1/微分方程数值解 - Project1.py"
```

项目一的完整实验会遍历多组网格并运行多种迭代法，耗时较长。如果只想快速查看结果，可以减小 `main()` 中 `time_error_analysis` 的 `n_values` 范围。

运行项目二：

```bash
python "Project2/差分格式求解代码.py"
```

项目二会在当前目录生成 12 张 PNG 图片，仓库中已经保留了运行结果，放在 `Project2/数值结果/`。

### 字体说明

项目一的绘图代码原本使用 `times+simsun.ttf` 设置中英文字体。仓库没有附带这个字体文件，代码会在字体不存在时自动回退到系统字体。如果图中的中文显示为方块，可以安装一款中文字体，或者把 `times+simsun.ttf` 放到脚本同一目录。

## 报告说明

- PDF 由对应的 DOCX 直接导出，保留原始公式、表格、图片和分页。
- Markdown 由同一份 DOCX 转换得到，公式使用 LaTeX，图片放在各项目的 `media/` 目录。
- 报告内容保留提交时的状态。代码只做了两处可运行性调整：字体缺失时回退，以及增加 `if __name__ == "__main__"` 入口。
- 本仓库仅用于课程学习记录和参考，请遵守学校的学术诚信规定。

## 许可证

本项目采用 MIT License，详见 [LICENSE](LICENSE)。
