import matplotlib.pyplot as plt
from matplotlib import animation

import os


def display_frames_as_gif(frames, name):
	patch = plt.imshow(frames[0])
	plt.axis('off')
	
	anim = animation.FuncAnimation(plt.gcf(), lambda i: patch.set_data(frames[i]), frames=len(frames), interval=1)
	anim.save(f'./render_{name}.gif', fps=30)


def saving_pic(x, y, fig_name, image_name, y_label, x_label="Iteration Step"):
	saving_pic_multi_line(x, (y,), fig_name, image_name, ("",), y_label, x_label, False)


def saving_pic_multi_line(x, y, fig_name, image_file_name, line_labels, y_label, x_label="Iteration Step", legend=True):
	plt.figure(figsize=(8, 6))
	for data, name in zip(y, line_labels):
		plt.plot(x, data, marker=None, linestyle='-', label=name)
	if legend: plt.legend()
	plt.title(fig_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(os.path.join(f'./{image_file_name}.png'))


def show_line_and_area(x, y, fig_name, image_file_name, y_label="Value", x_label="Iteration Step", legend=True):
	# min, max, avg
	# 绘制统计数据的变化趋势
	plt.figure(figsize=(10, 6))
	plt.plot(y[2], label='Mean', color='red')
	
	# 使用 fill_between 填充最小值和最大值之间的区域
	plt.fill_between(x, y[0], y[1], color='gray', alpha=0.2,
	                 label=f'Range')
	
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(fig_name)
	# 添加图例
	if legend: plt.legend()
	
	plt.savefig(os.path.join(f'./{image_file_name}.png'))
