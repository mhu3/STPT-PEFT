import matplotlib.pyplot as plt

# 数据
methods = ['Finetune', 'Prompt', 'LoRA', 'Houlsby', 'Bitfit', 'AdapterBias','Weighted-sum']
params = [8*10**5, 2*10**4, 5.7*10**4, 1.4*10**5, 1.6*10**4, 2*10**3, 8]  
accuracy = [0.7759, 0.6509, 0.7119, 0.7233, 0.6671, 0.6174, 0.6593]

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