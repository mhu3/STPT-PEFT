import matplotlib.pyplot as plt

# 数据
methods = ['Finetune', 'Prompt', 'LoRA', 'Houlsby', 'Bitfit', 'Adapterbias', 'Weighted-sum']
params = [1*10**6, 2.6*10**4, 5*10**4, 1.4*10**5, 1*10**4, 2*10**3, 8]  
accuracy = [0.83125, 0.60625, 0.8125, 0.76875, 0.71575, 0.66875, 0.63125]

# 创建一个图形和轴
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制数据点
for i, method in enumerate(methods):
    if method == 'Finetune':
        ax.scatter(params[i], accuracy[i], color='red', label=method)
    else:
        ax.scatter(params[i], accuracy[i], color='black', label=method)

# 添加每个点的标签
for i, method in enumerate(methods):
    ax.annotate(method, (params[i], accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# 设置轴标签和标题
ax.set_xlabel('Parameters')
ax.set_ylabel('Accuracy')
ax.set_title('Trajectory Classification')

# 设置x轴为对数刻度
ax.set_xscale('log')

# 显示图例
ax.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# Save the plot
plt.savefig('comparison.png')