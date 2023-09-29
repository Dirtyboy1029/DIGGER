# -*- coding: utf-8 -*- 
# @Time : 2023/9/26 20:47 
# @Author : DirtyBoy 
# @File : merge_dis2distribution_plot.py
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from Digger.utils import remove_outliers
import numpy as np
from Digger.experiments.distribution_diff_plot import read_data
import matplotlib.pyplot as plt

legend_font = {
    # 'family': 'Arial',  # 字体
    # 'style': 'normal',
    'size': 8,  # 字号
    'weight': "bold",  # 是否加粗，不加粗
}

def merge2distribution(dis, source_distribution: list):
    source_distribution = remove_outliers(source_distribution)
    dis_arr = np.array([dis] * len(source_distribution))
    new_distribution = dis_arr + np.array(source_distribution)
    return new_distribution


if __name__ == '__main__':
    experiment = 'seen'
    seen_benchmark_learned, seen_benchmark_unlearned, seen_test_learned, \
    seen_test_unlearned, seen_union_learned, seen_union_unlearned = read_data(experiment)

    experiment = 'unseen'
    unseen_benchmark_learned, unseen_benchmark_unlearned, unseen_test_learned, \
    unseen_test_unlearned, unseen_union_learned, unseen_union_unlearned = read_data(experiment)

    experiment = 'mayseen'
    mayseen_benchmark_learned, mayseen_benchmark_unlearned, mayseen_test_learned, \
    mayseen_test_unlearned, mayseen_union_learned, mayseen_union_unlearned = read_data(experiment)



    fig, axes = plt.subplots(2, 3, figsize=(8, 6), )
    # 设置整个图形的背景颜色
    # fig.patch.set_facecolor('#f2f2f2')
    # 绘制子图1
    ax1 = axes[0, 0]
    ax1.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax7.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    ax1.hist(remove_outliers(unseen_benchmark_learned), bins=70, color='green', density=True, alpha=0.5, )
    ax1.patch.set_facecolor('lightgray')
    # 添加子图1的网格
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axes[0, 1]
    ax2.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax8.hist(mayseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    # ax8.hist(remove_outliers(benchmark_unlearned), bins=70, color='red', density=True, alpha=0.5, )
    ax2.hist(remove_outliers(unseen_benchmark_learned), bins=70, color='green', density=True, alpha=0.5, )
    ax2.patch.set_facecolor('lightgray')
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = axes[0, 2]
    ax3.hist(unseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax9.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    ax3.hist(remove_outliers(unseen_benchmark_learned), bins=70, color='green', density=True, alpha=0.5, )
    ax3.patch.set_facecolor('lightgray')
    ax3.legend(['testset', 'bar-learn'], prop=legend_font)
    ax3.grid(True, linestyle='--', alpha=0.6)
    unseen_dis = wasserstein_distance(unseen_union_unlearned, unseen_test_unlearned)
    seen_dis = wasserstein_distance(seen_union_unlearned, seen_test_unlearned)
    mayseen_dis = wasserstein_distance(mayseen_union_unlearned, mayseen_test_unlearned)

    new_unseen_bar_learned = merge2distribution(unseen_dis,unseen_benchmark_learned)
    new_seen_bar_learned = merge2distribution(seen_dis, seen_benchmark_learned)
    new_mayseen_bar_learned = merge2distribution(seen_dis, mayseen_benchmark_learned)

    ax4 = axes[1, 0]
    ax4.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax7.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    ax4.hist(new_seen_bar_learned, bins=70, color='green', density=True, alpha=0.5, )
    ax4.patch.set_facecolor('lightgray')
    # 添加子图1的网格
    ax4.grid(True, linestyle='--', alpha=0.6)

    ax5 = axes[1, 1]
    ax5.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax8.hist(mayseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    # ax8.hist(remove_outliers(benchmark_unlearned), bins=70, color='red', density=True, alpha=0.5, )
    ax5.hist(new_mayseen_bar_learned, bins=70, color='green', density=True, alpha=0.5, )
    ax5.patch.set_facecolor('lightgray')
    ax5.grid(True, linestyle='--', alpha=0.6)
    #
    ax6 = axes[1, 2]
    ax6.hist(unseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax9.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    ax6.hist(new_unseen_bar_learned, bins=70, color='green', density=True, alpha=0.5, )
    ax6.patch.set_facecolor('lightgray')
    ax6.legend(['testset', 'bar-learn'], prop=legend_font)
    ax6.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
