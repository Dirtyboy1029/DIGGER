# -*- coding: utf-8 -*- 
# @Time : 2023/9/28 11:05 
# @Author : DirtyBoy 
# @File : demo1.py
from real_world.experiments.utils import read_pkl, remove_outliers_, merge2distribution
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
model_type = 'GPT2_XL'
br = read_pkl('intermediate_file/' + model_type + '/book_experiment.pkl').reshape((500, -1)).squeeze()
qr = read_pkl('intermediate_file/' + model_type + '/quote_experiment.pkl').reshape((500, -1)).squeeze()

be = read_pkl('intermediate_file/' + model_type + '/book_real_world.pkl').reshape((500, -1)).squeeze()
qe = read_pkl('intermediate_file/' + model_type + '/quote_real_world.pkl').reshape((500, -1)).squeeze()

ben_pre = read_pkl('intermediate_file/' + model_type + '/pre.pkl')[0]
ben_tune = read_pkl('intermediate_file/' + model_type + '/tune.pkl')[0]
benchmark_learned = []
benchmark_unlearned = []
for i in range(len(ben_pre)):
    benchmark_learned.append(ben_pre[i][2])
    benchmark_unlearned.append(ben_tune[i][2])
benchmark_learned = np.array(benchmark_learned)
benchmark_unlearned = np.array(benchmark_unlearned)
dis = -wasserstein_distance(br, be)
new_bar_learned = merge2distribution(dis, list(benchmark_learned))
bar_mean = np.mean(new_bar_learned)
bar_std = np.std(new_bar_learned)
# conf = []
# for i, item in enumerate(qr):
#     probability = norm.cdf(item, loc=bar_mean, scale=bar_std)
#     conf.append(probability)
# np.save('gpt2.npz',conf)
# print(conf)
# plt.hist(conf, bins=70, color='blue', density=True, alpha=0.2, )
# plt.show()
# fig, axes = plt.subplots(2, 5, figsize=(20, 4))




fig = plt.figure(figsize=(14, 7), )
gs = GridSpec(2, 1, height_ratios=[4.7, 0.3])
axes = gs[0].subgridspec(2, 5)
ax1 = plt.subplot(axes[0, 1])
ax1.hist(qe, bins=20, color='yellow', density=True, alpha=0.2, )
ax1.hist(be, bins=20, color='green', density=True, alpha=0.2, )

ax1.set_yticks([0,1,2,3,4,5])
ax1.set_title('Reference-Tuned', fontweight='bold', fontsize=16)
# ax1.set_xticks([0, 1, 2,])

ax1.patch.set_facecolor('lightgray')
ax1.grid(True, linestyle='--', alpha=0.6)

ax4 = plt.subplot(axes[0, 0])
ax4.hist(remove_outliers_(qr), bins=20, color='blue', density=True, alpha=0.2, )
ax4.hist(br, bins=20, color='red', density=True, alpha=0.2, )
ax4.set_title('Vanilla-Tuned', fontweight='bold', fontsize=16)
ax4.set_yticks([0,1,2,3,4,5])
ax4.set_xlim(0, 1.5)
ax4.set_ylabel('GPT2 XL', fontweight='bold', fontsize=20)

ax4.patch.set_facecolor('lightgray')
ax4.grid(True, linestyle='--', alpha=0.6)

ax7 = plt.subplot(axes[0, 2])
ax7.hist(remove_outliers_(qr), bins=20, color='blue', density=True, alpha=0.2, )
ax7.set_title('First comparison', fontweight='bold', fontsize=16)
ax7.hist(benchmark_learned, bins=20, color='orange', density=True, alpha=0.5, )
ax7.set_yticks([0,1,2,3,4,5])
ax7.set_xlim(0, 1.5)
ax7.patch.set_facecolor('lightgray')
ax7.grid(True, linestyle='--', alpha=0.6)

ax10 = plt.subplot(axes[0, 3])
ax10.hist(br, bins=20, color='red', density=True, alpha=0.2, )
# ax7.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, ) Calculate distance
ax10.hist(be, bins=20, color='green', density=True, alpha=0.2, )
ax10.set_title('Calculate distance', fontweight='bold', fontsize=16)
ax10.set_yticks([0,1,2,3,4,5])
# ax10.set_xticks([0, 1, 2, 3])
ax10.patch.set_facecolor('lightgray')
ax10.grid(True, linestyle='--', alpha=0.6)
ax10.text(0.25, 2.8, 'Dis=' + str(round(wasserstein_distance(br, be), 3)),
          fontsize=15, color='black', fontweight='bold')
dis = -wasserstein_distance(br, be)
new_bar_learned = merge2distribution(dis, list(benchmark_learned))
ax13 = plt.subplot(axes[0, 4])
ax13.hist(remove_outliers_(qr), bins=20, color='blue', density=True, alpha=0.2, )
ax13.set_title('Actual comparison', fontweight='bold', fontsize=16)
ax13.hist(new_bar_learned, bins=20, color='orange', density=True, alpha=0.5, )
ax13.set_yticks([0,1,2,3,4,5])
ax13.set_xlim(0, 1.5)
ax13.patch.set_facecolor('lightgray')
ax13.grid(True, linestyle='--', alpha=0.6)





model_type = 'LLaMA_7b'
br = read_pkl('intermediate_file/' + model_type + '/book_experiment.pkl').reshape((500, -1)).squeeze()
qr = read_pkl('intermediate_file/' + model_type + '/quote_experiment.pkl').reshape((500, -1)).squeeze()

be = read_pkl('intermediate_file/' + model_type + '/book_real_world.pkl').reshape((500, -1)).squeeze()
qe = read_pkl('intermediate_file/' + model_type + '/quote_real_world.pkl').reshape((500, -1)).squeeze()

ben_pre = read_pkl('intermediate_file/' + model_type + '/pre.pkl')[0]
ben_tune = read_pkl('intermediate_file/' + model_type + '/tune.pkl')[0]
benchmark_learned = []
benchmark_unlearned = []
for i in range(len(ben_pre)):
    benchmark_learned.append(ben_pre[i][2])
    benchmark_unlearned.append(ben_tune[i][2])
benchmark_learned = np.array(benchmark_learned)
benchmark_unlearned = np.array(benchmark_unlearned)





ax2 = plt.subplot(axes[1, 1])
ax2.hist(qe, bins=20, color='yellow', density=True, alpha=0.2, )
ax2.hist(be, bins=20, color='green', density=True, alpha=0.2, )
ax2.set_ylim(0, 5)
ax2.set_xlim(0., 1.0)
# ax2.set_xticks([0.0, 0.5, 1.0])

ax2.patch.set_facecolor('lightgray')
ax2.grid(True, linestyle='--', alpha=0.6)

ax5 = plt.subplot(axes[1, 0])
ax5.hist(qr, bins=20, color='blue', density=True, alpha=0.2, )
ax5.hist(br, bins=20, color='red', density=True, alpha=0.2, )
ax5.patch.set_facecolor('lightgray')
ax5.set_ylabel('LLaMA 7b', fontweight='bold', fontsize=20)
ax5.set_ylim(0, 5)
ax5.set_xlim(0, 1.5)
ax5.grid(True, linestyle='--', alpha=0.6)

ax8 = plt.subplot(axes[1, 2])
ax8.hist(qr, bins=20, color='blue', density=True, alpha=0.2, )
ax8.hist(benchmark_learned, bins=20, color='orange', density=True, alpha=0.5, )
ax8.set_ylim(0, 5)
ax8.set_xlim(0, 1.5)
# ax8.set_xticks([0.0, 0.5, 1.0])
ax8.patch.set_facecolor('lightgray')
ax8.grid(True, linestyle='--', alpha=0.6)

ax11 = plt.subplot(axes[1, 3])
ax11.hist(br, bins=20, color='red', density=True, alpha=0.2, )
ax11.hist(be, bins=20, color='green', density=True, alpha=0.2, )
ax11.set_ylim(0, 5)
ax11.set_xlim(0., 1.0)
ax11.patch.set_facecolor('lightgray')
ax11.grid(True, linestyle='--', alpha=0.6)
ax11.text(0.15, 4,
          'Dis=' + str(round(wasserstein_distance(br, be), 3)),
          fontsize=15, color='black', fontweight='bold')
dis = -wasserstein_distance(br, be)
new_bar_learned = merge2distribution(dis, list(benchmark_learned))
ax14 = plt.subplot(axes[1, 4])
ax14.hist(qr, bins=20, color='blue', density=True, alpha=0.2, )
ax14.hist(new_bar_learned, bins=20, color='orange', density=True, alpha=0.5, )
ax14.set_ylim(0, 5)
ax14.set_xlim(0, 1.5)
# ax14.set_xticks([0.0, 0.5, 1.0])
ax14.patch.set_facecolor('lightgray')
ax14.grid(True, linestyle='--', alpha=0.6)

for i in range(2):
    for j in range(5):
        ax = plt.subplot(axes[i, j])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

legend_ax = plt.subplot(gs[1])

x = [1.8, 2.4, 3, 3.6, 4.2]
y = [0, 0, 0, 0, 0]
labels = ['target dataset', 'unlearned dataset II', 'target dataset', 'unlearned dataset II',
          'unlearned dataset I ']
label_offset = 0.3
# 创建散点图
legend_ax.scatter([x[0]], [y[0]], s=500, c='yellow', alpha=0.2)
legend_ax.scatter([x[1]], [y[1]], s=500, c='green', alpha=0.2)
legend_ax.scatter([x[2]], [y[2]], s=500, c='blue', alpha=0.2)
legend_ax.scatter([x[3]], [y[3]], s=500, c='red', alpha=0.2)
legend_ax.scatter([x[4]], [y[4]], s=500, c='orange', alpha=0.5)

legend_ax.axis('off')

# 添加标注
for i in range(len(x)):
    legend_ax.text(x[i] - 0.015, y[i] - 0.15, labels[i], fontsize=12, fontweight='bold', ha='center',
                   va='bottom')

plt.tight_layout()
plt.savefig('real_world.pdf')

plt.show()
