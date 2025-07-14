from tabulate import tabulate

# 表格数据
data = [
    ["分类", "线弹性力学（小变形理论）", "连续介质力学（有限变形理论）"],
    ["基本假设",
     "1. 小变形（位移梯度≪1）<br>2. 材料均匀且各向同性<br>3. 静态平衡或低速动态",
     "1. 大变形（任意位移梯度）<br>2. 可处理非均匀、各向异性材料<br>3. 动态非线性效应显著"],
    ["几何关系",
     "线性应变张量：<br>\\[ \\boldsymbol{\\varepsilon} = \\frac{1}{2}(\\nabla \\mathbf{u} + \\nabla \\mathbf{u}^T) \\]<br>（忽略高阶项）",
     "有限应变张量（Green-Lagrange）：<br>\\[ \\mathbf{E} = \\frac{1}{2}(\\mathbf{F}^T\\mathbf{F} - \\mathbf{I}) \\]<br>变形梯度：<br>\\[ \\mathbf{F} = \\frac{\\partial \\mathbf{x}}{\\partial \\mathbf{X}} \\]"],
    ["材料关系",
     "胡克定律（线性）：<br>\\[ \\boldsymbol{\\sigma} = \\mathbb{C}:\\boldsymbol{\\varepsilon} \\]<br>（弹性张量 \\(\\mathbb{C}\\) 为常数）",
     "超弹性本构关系（非线性）：<br>\\[ \\boldsymbol{\\sigma} = \\frac{1}{J} \\frac{\\partial W}{\\partial \\mathbf{F}} \\mathbf{F}^T \\]<br>（\\(W\\): 应变能函数，\\(J = \\det \\mathbf{F}\\)）"],
    ["平衡关系",
     "线性平衡方程：<br>\\[ \\nabla \\cdot \\boldsymbol{\\sigma} + \\mathbf{b} = \\mathbf{0} \\]<br>（未变形构型下求解）",
     "非线性平衡方程：<br>\\[ \\nabla \\cdot \\mathbf{P} + \\mathbf{b} = \\rho \\ddot{\\mathbf{u}} \\]<br>（\\(\\mathbf{P}\\): 第一Piola-Kirchhoff应力）"],
    ["变形描述",
     "未变形构型与变形构型近似重合（无需区分）",
     "严格区分<b>参考构型</b>（\\(\\mathbf{X}\\)）和<b>当前构型</b>（\\(\\mathbf{x}\\)）"],
    ["典型应用",
     "桥梁、机械零件等小变形问题",
     "橡胶、生物组织、布料等大变形问题；MeshGraphNets中的自适应仿真"],
    ["计算复杂度",
     "低（线性方程组）",
     "高（非线性迭代，需跟踪变形历史）"]
]

# 生成HTML表格
html_table = tabulate(data, headers="firstrow", tablefmt="html")

# 添加CSS样式和MathJax支持（用于渲染LaTeX公式）
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>线弹性力学 vs. 连续介质力学对比</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: Arial, sans-serif;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>线弹性力学 vs. 连续介质力学对比</h1>
    {html_table}
</body>
</html>
"""

# 保存为HTML文件
with open("mechanics_comparison.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML文件已生成：mechanics_comparison.html")