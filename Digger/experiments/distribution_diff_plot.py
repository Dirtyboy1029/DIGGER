# -*- coding: utf-8 -*- 
# @Time : 2023/9/24 5:36 
# @Author : DirtyBoy 
# @File : distribution_diff_plot.py
from Digger.utils import read_pkl, remove_outliers
import numpy as np
import scipy
import matplotlib.pyplot as plt
# from scipy.spatial.distance import hellinger
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from Digger.experiments.acc_f1 import evaluate

def merge2distribution(dis, source_distribution: list):
    source_distribution = remove_outliers(source_distribution)
    dis_arr = np.array([dis] * len(source_distribution))
    new_distribution = dis_arr + np.array(source_distribution)
    return new_distribution


def Kullback_Leibler(a, b):
    ##KL散度（Kullback-Leibler散度）
    return scipy.stats.entropy(a, b)


def get_auc(y_label, y_pre):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)
    return auc(fpr, tpr)


def read_data(experiment):
    benchmark_learned = read_pkl('intermediate_file/' + experiment + '/benchmark_learned.pkl').reshape(
        (7000, -1)).squeeze()
    benchmark_unlearned = read_pkl('intermediate_file/' + experiment + '/benchmark_unlearned.pkl').reshape(
        (7000, -1)).squeeze()
    union_unlearned = read_pkl('intermediate_file/' + experiment + '/union_unlearned.pkl')
    union_unlearned = union_unlearned.reshape((len(union_unlearned) * 200, -1)).squeeze()
    test_unlearned = read_pkl('intermediate_file/' + experiment + '/test_unlearned.pkl')
    test_unlearned = test_unlearned.reshape((len(test_unlearned) * 200, -1)).squeeze()
    if experiment == 'mayseen':
        unlearned_book_num = [1, 2, 4, 6, 10, 11, 12]
        learned_book_num = [0, 3, 5, 7, 8, 9, 13]

        learned_num = []
        unlearned_num = []
        for i in unlearned_book_num:
            unlearned_num = unlearned_num + list(range(i * 200, (i + 1) * 200))
        for i in learned_book_num:
            learned_num = learned_num + list(range(i * 200, (i + 1) * 200))

        vanilla_test_learned = read_pkl('intermediate_file/mayseen/vanilla/test_learned.pkl')
        vanilla_test_learned = vanilla_test_learned.reshape((len(vanilla_test_learned) * 200, -1)).squeeze()
        target_test_learned = read_pkl('intermediate_file/mayseen/target/test_learned.pkl')
        target_test_learned = target_test_learned.reshape((len(target_test_learned) * 200, -1)).squeeze()

        vanilla_union_learned = read_pkl('intermediate_file/mayseen/vanilla/union_learned.pkl')
        vanilla_union_learned = vanilla_union_learned.reshape((len(vanilla_union_learned) * 200, -1)).squeeze()
        target_union_learned = read_pkl('intermediate_file/mayseen/target/union_learned.pkl')
        target_union_learned = target_union_learned.reshape((len(target_union_learned) * 200, -1)).squeeze()

        test_learned = np.array(list(target_test_learned[learned_num]) + list(vanilla_test_learned[unlearned_num]))
        union_learned = np.array(list(target_union_learned[learned_num]) + list(vanilla_union_learned[unlearned_num]))
    else:
        test_learned = read_pkl('intermediate_file/' + experiment + '/test_learned.pkl')
        test_learned = test_learned.reshape((len(test_learned) * 200, -1)).squeeze()
        union_learned = read_pkl('intermediate_file/' + experiment + '/union_learned.pkl')
        union_learned = union_learned.reshape((len(union_learned) * 200, -1)).squeeze()

    return benchmark_learned, benchmark_unlearned, test_learned, test_unlearned, union_learned, union_unlearned


if __name__ == '__main__':
    experiment = 'unseen'
    benchmark_learned, benchmark_unlearned, test_learned, test_unlearned, union_learned, union_unlearned = read_data(
        experiment)
    legend_font = {
        # 'family': 'Arial',  # 字体
        # 'style': 'normal',
        'size': 15,  # 字号
        'weight': "bold",  # 是否加粗，不加粗
    }
    experiment_type = 'plot_differnt_env'

    if experiment_type == 'plot_fig':
        '''
        experiment = 'seen'
        seen_benchmark_learned, seen_benchmark_unlearned, seen_test_learned, \
        seen_test_unlearned, seen_union_learned, seen_union_unlearned = read_data(experiment)

        experiment = 'unseen'
        unseen_benchmark_learned, unseen_benchmark_unlearned, unseen_test_learned, \
        unseen_test_unlearned, unseen_union_learned, unseen_union_unlearned = read_data(experiment)

        experiment = 'mayseen'
        mayseen_benchmark_learned, mayseen_benchmark_unlearned, mayseen_test_learned, \
        mayseen_test_unlearned, mayseen_union_learned, mayseen_union_unlearned = read_data(experiment)

        legend_list = ['unseen', 'seen']
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(12, 4), )##sharex=True, sharey=True
        # 设置整个图形的背景颜色
        # fig.patch.set_facecolor('#f2f2f2')
        # 绘制子图1
        ax1 = axes[0, 0]
        ax1.hist(seen_test_learned, bins=70, color='green', density=True, alpha=0.2, )
        ax1.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax1.set_title('Benchmark')
        ax1.set_xlabel('Loss Gap')
        ax1.set_ylabel('Probability Density')
        ax1.legend(legend_list)
        # 设置子图1的背景颜色
        ax1.patch.set_facecolor('lightgray')
        # 添加子图1的网格
        ax1.grid(True, linestyle='--', alpha=0.6)
        # 绘制子图2
        ax2 = axes[0, 1]
        ax2.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax2.hist(mayseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax2.set_title('Test Samples in the Experimental Environment')
        ax2.set_xlabel('Loss Gap')
        ax2.set_ylabel('Probability Density')
        ax2.legend(legend_list)
        # 设置子图2的背景颜色
        ax2.patch.set_facecolor('lightgray')
        # 添加子图2的网格
        ax2.grid(True, linestyle='--', alpha=0.6)
        # 绘制子图3
        ax3 = axes[0, 2]
        ax3.hist(unseen_test_learned, bins=70, color='green', density=True, alpha=0.2, )
        ax3.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax3.set_title('Test Samples in the Real World')
        ax3.set_xlabel('Loss Gap')
        ax3.set_ylabel('Probability Density')
        ax3.legend(legend_list)
        # 设置子图3的背景颜色
        ax3.patch.set_facecolor('lightgray')
        # 添加子图3的网格
        ax3.grid(True, linestyle='--', alpha=0.6)
        # 调整子图之间的间距
        plt.tight_layout()
        # 显示图形
        plt.savefig('Figure/' + experiment + '_verification_experiment.pdf')
        plt.show()
        '''
        benchmark_unlearned = remove_outliers(benchmark_unlearned)
        benchmark_learned = remove_outliers(benchmark_learned)
        auc_ = get_auc([1] * len(benchmark_learned) + [0] * len(benchmark_unlearned),
                       benchmark_learned + benchmark_unlearned)

        fig = plt.figure(figsize=(8, 6))

        plt.hist(remove_outliers(benchmark_learned), bins=90, color='blue', density=True, alpha=0.2, )
        plt.hist(remove_outliers(benchmark_unlearned), bins=90, color='red', density=True, alpha=0.2, )

        plt.title('Histogram Example')

        plt.xlabel('Samples Loss Gap')
        plt.ylabel('Probability Density')

        plt.legend(['learned', 'unlearned'], loc='upper right')
        ax = plt.subplot(111)
        ax.patch.set_facecolor('lightgray')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.text(0.6, 0.8, 'AUC=' + str(auc_), fontweight='bold', fontsize=15)

        plt.show()
    elif experiment_type == 'experiments':
        # ##KL散度（Kullback-Leibler散度）
        # print(scipy.stats.entropy(union_unlearned, test_unlearned))
        # print(scipy.stats.entropy(test_unlearned, union_unlearned))
        #
        # print(scipy.stats.entropy(union_learned, test_learned))
        # print(scipy.stats.entropy(test_learned, union_learned))
        #
        # print(scipy.stats.entropy(benchmark_learned, benchmark_unlearned))
        # print(scipy.stats.entropy(union_learned, union_unlearned))
        # print(scipy.stats.entropy(test_learned, test_unlearned))

        ##Jensen-Shannon散度：Jensen-Shannon散度是KL散度的一种变体
        # print(jensenshannon(union_unlearned, test_unlearned))
        # print(jensenshannon(test_unlearned, union_unlearned))
        #
        # print(jensenshannon(union_learned, test_learned))
        # print(jensenshannon(test_learned, union_learned))
        #
        # print(jensenshannon(benchmark_learned, benchmark_unlearned))
        # print(jensenshannon(union_learned, union_unlearned))
        # print(jensenshannon(test_learned, test_unlearned))

        ###Earth Mover's Distance：这是一种度量两个分布之间差异的方法
        print(wasserstein_distance(union_unlearned, test_unlearned))
        print(wasserstein_distance(test_unlearned, union_unlearned))

        print(wasserstein_distance(union_learned, test_learned))
        print(wasserstein_distance(test_learned, union_learned))

        print(wasserstein_distance(benchmark_learned, benchmark_unlearned))
        print(wasserstein_distance(union_learned, union_unlearned))
        print(wasserstein_distance(test_learned, test_unlearned))
    elif experiment_type == 'plot_differnt_env':
        experiment = 'seen'
        seen_benchmark_learned, seen_benchmark_unlearned, seen_test_learned, \
        seen_test_unlearned, seen_union_learned, seen_union_unlearned = read_data(experiment)

        experiment = 'unseen'
        unseen_benchmark_learned, unseen_benchmark_unlearned, unseen_test_learned, \
        unseen_test_unlearned, unseen_union_learned, unseen_union_unlearned = read_data(experiment)

        experiment = 'mayseen'
        mayseen_benchmark_learned, mayseen_benchmark_unlearned, mayseen_test_learned, \
        mayseen_test_unlearned, mayseen_union_learned, mayseen_union_unlearned = read_data(experiment)

        unseen_dis = wasserstein_distance(unseen_union_unlearned, unseen_test_unlearned)
        seen_dis = -wasserstein_distance(seen_union_unlearned, seen_test_unlearned)
        mayseen_dis = -wasserstein_distance(mayseen_union_unlearned, mayseen_test_unlearned)

        new_unseen_bar_learned = merge2distribution(unseen_dis, unseen_benchmark_learned)
        new_seen_bar_learned = merge2distribution(seen_dis, seen_benchmark_learned)
        new_mayseen_bar_learned = merge2distribution(seen_dis, mayseen_benchmark_learned)

        # fig, axes = plt.subplots(3, 5, figsize=(16, 12), )
        fig = plt.figure(figsize=(16, 12), )
        gs = GridSpec(2, 1, height_ratios=[4.7, 0.3])
        axes = gs[0].subgridspec(3, 5)
        ax1 = plt.subplot(axes[0, 1])
        ax1.hist(seen_union_learned, bins=70, color='yellow', density=True, alpha=0.2, )
        ax1.hist(seen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax1.set_title('Reference-Tuned', fontweight='bold', fontsize=18)
        ax1.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        ax1.patch.set_facecolor('lightgray')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax4 = plt.subplot(axes[0, 0])
        ax4.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax4.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax4.set_title('Vanilla-Tuned', fontweight='bold', fontsize=18)
        ax4.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ax4.set_ylabel('Seen', fontweight='bold', fontsize=18)
        # ax4.set_xticks([0, 1, 2, 3])
        ax4.patch.set_facecolor('lightgray')
        ax4.grid(True, linestyle='--', alpha=0.6)
        # ax4.text(0.25, 2.8,
        #          'AUC=' + str(round(get_auc(y_label=[0] * 3000 + [1] * 3000, y_pre=list(seen_test_unlearned) + list(seen_test_learned)),3)),
        #          fontsize=15, color='black', fontweight='bold')

        ax7 = plt.subplot(axes[0, 2])
        ax7.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax7.set_title('First comparison', fontweight='bold', fontsize=18)
        ax7.hist(remove_outliers(benchmark_learned), bins=70, color='orange', density=True, alpha=0.5, )
        ax7.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        # ax7.set_xticks([0, 1, 2, 3])
        ax7.patch.set_facecolor('lightgray')
        ax7.grid(True, linestyle='--', alpha=0.6)

        ax10 = plt.subplot(axes[0, 3])
        ax10.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        # ax7.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, ) Calculate distance
        ax10.hist(seen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax10.set_title('Calculate distance', fontweight='bold', fontsize=18)
        ax10.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        # ax10.set_xticks([0, 1, 2, 3])
        ax10.patch.set_facecolor('lightgray')
        ax10.grid(True, linestyle='--', alpha=0.6)
        ax10.text(0.25, 2.8, 'Dis=' + str(round(wasserstein_distance(seen_test_unlearned, seen_union_unlearned), 3)),
                  fontsize=15, color='black', fontweight='bold')

        ax13 = plt.subplot(axes[0, 4])
        ax13.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax13.set_title('Actual comparison', fontweight='bold', fontsize=18)
        ax13.hist(new_seen_bar_learned, bins=70, color='orange', density=True, alpha=0.5, )
        ax13.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        # ax13.set_xticks([0, 1, 2, 3])
        ax13.patch.set_facecolor('lightgray')
        ax13.grid(True, linestyle='--', alpha=0.6)

        ax2 = plt.subplot(axes[1, 1])
        ax2.hist(mayseen_union_learned, bins=70, color='yellow', density=True, alpha=0.2, )
        ax2.hist(mayseen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax2.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        ax2.patch.set_facecolor('lightgray')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax5 = plt.subplot(axes[1, 0])


        # unlearned_book_num = [1, 2, 4, 6, 10, 11, 12]
        # learned_book_num = [0, 3, 5, 7, 8, 9, 13]
        #
        # learned_num = []
        # unlearned_num = []
        # for i in unlearned_book_num:
        #     unlearned_num = unlearned_num + list(range(i * 200, (i + 1) * 200))
        # for i in learned_book_num:
        #     learned_num = learned_num + list(range(i * 200, (i + 1) * 200))

        # test_learned = np.array(list(mayseen_test_learned[learned_num]) + list(mayseen_test_learned[unlearned_num]))
        # union_learned = np.array(list(mayseen_union_learned[learned_num]) + list(mayseen_union_learned[unlearned_num]))

        # ax5.hist(mayseen_test_learned[learned_num], bins=70, color='blue', density=True, alpha=0.2, )
        # ax5.hist(mayseen_test_learned[unlearned_num], bins=70, color='red', density=True, alpha=0.2, )

        ax5.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax5.hist(mayseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax5.patch.set_facecolor('lightgray')
        ax5.set_ylabel('Possibly seen', fontweight='bold', fontsize=18)
        ax5.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ax5.grid(True, linestyle='--', alpha=0.6)
        # ax5.text(0.25, 2.8,
        #          'AUC=' + str(round(get_auc(y_label=[0] * 2800 + [1] * 2800,
        #                                     y_pre=list(mayseen_test_unlearned) + list(mayseen_test_learned)), 3)),
        #          fontsize=15, color='black', fontweight='bold')

        ax8 = plt.subplot(axes[1, 2])
        ax8.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax8.hist(remove_outliers(benchmark_learned), bins=70, color='orange', density=True, alpha=0.5, )
        ax8.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ax8.patch.set_facecolor('lightgray')
        ax8.grid(True, linestyle='--', alpha=0.6)

        ax11 = plt.subplot(axes[1, 3])
        ax11.hist(mayseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax11.hist(mayseen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax11.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ax11.patch.set_facecolor('lightgray')
        ax11.grid(True, linestyle='--', alpha=0.6)
        ax11.text(0.25, 2.8,
                  'Dis=' + str(round(wasserstein_distance(mayseen_test_unlearned, mayseen_union_unlearned), 3)),
                  fontsize=15, color='black', fontweight='bold')

        ax14 = plt.subplot(axes[1, 4])
        ax14.hist(mayseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax14.hist(new_mayseen_bar_learned, bins=70, color='orange', density=True, alpha=0.5, )
        ax14.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        ax14.patch.set_facecolor('lightgray')
        ax14.grid(True, linestyle='--', alpha=0.6)

        ax3 = plt.subplot(axes[2, 1])
        ax3.hist(unseen_union_learned, bins=70, color='yellow', density=True, alpha=0.2, )
        ax3.hist(unseen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax3.set_yticks([0, 1, 2, 3, 4, 5, 6, ])

        ax3.patch.set_facecolor('lightgray')
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax6 = plt.subplot(axes[2, 0])
        ax6.hist(unseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax6.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax6.set_yticks([0, 1, 2, 3, 4, 5, 6, ])
        ax6.patch.set_facecolor('lightgray')
        ax6.set_ylabel('Unseen', fontweight='bold', fontsize=20)
        # ax6.legend(['testset', 'labeled unlearn'], prop=legend_font)
        ax6.grid(True, linestyle='--', alpha=0.6)

        ax9 = plt.subplot(axes[2, 2])
        ax9.hist(unseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax9.hist(remove_outliers(benchmark_learned), bins=70, color='orange', density=True, alpha=0.5, )
        ax9.set_yticks([0, 1, 2, 3, 4, 5, 6, ])
        ax9.patch.set_facecolor('lightgray')
        ax9.grid(True, linestyle='--', alpha=0.6)

        ax12 = plt.subplot(axes[2, 3])
        ax12.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        ax12.hist(unseen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax12.patch.set_facecolor('lightgray')
        ax12.set_yticks([0, 1, 2, 3, 4, 5, 6, ])
        # ax12.legend(['unlearn-test', 'unlearn-exp'], prop=legend_font)
        ax12.grid(True, linestyle='--', alpha=0.6)
        ax12.text(0.25, 2.8,
                  'Dis=' + str(round(wasserstein_distance(unseen_test_unlearned, unseen_union_unlearned), 3)),
                  fontsize=15, color='black', fontweight='bold')

        ax15 = plt.subplot(axes[2, 4])
        ax15.hist(unseen_test_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax15.hist(new_unseen_bar_learned, bins=70, color='orange', density=True, alpha=0.5, )
        ax15.patch.set_facecolor('lightgray')
        ax15.set_yticks([0, 1, 2, 3, 4, 5, 6, ])
        # ax15.legend(['testset', 'bar-learn'], prop=legend_font)
        ax15.grid(True, linestyle='--', alpha=0.6)
        for i in range(3):
            for j in range(5):
                ax = plt.subplot(axes[i, j])
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')

        # for ax in axes.flat:
        #     # 获取x轴和y轴的刻度标签
        #     x_tick_labels = ax.get_xticklabels()
        #     y_tick_labels = ax.get_yticklabels()
        #
        #     # 设置刻度字体为粗体
        #     for label in x_tick_labels + y_tick_labels:
        #         label.set_weight('bold')
        #         label.set_fontsize(10)

        # fig.text(0.5, 0.04, 'x', ha='center', va='center', fontsize=20)
        # fig.text(-0.01, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=20)
        legend_ax = plt.subplot(gs[1])

        x = [1, 2, 3, 4, 5]
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
            legend_ax.text(x[i] - 0.05, y[i] - 0.08, labels[i], fontsize=12, fontweight='bold', ha='center',
                           va='bottom')

        plt.tight_layout()
        plt.savefig('111.pdf')
        plt.show()
    elif experiment_type == '111':
        experiment = 'seen'
        seen_benchmark_learned, seen_benchmark_unlearned, seen_test_learned, \
        seen_test_unlearned, seen_union_learned, seen_union_unlearned = read_data(experiment)

        experiment = 'unseen'
        unseen_benchmark_learned, unseen_benchmark_unlearned, unseen_test_learned, \
        unseen_test_unlearned, unseen_union_learned, unseen_union_unlearned = read_data(experiment)

        experiment = 'mayseen'
        mayseen_benchmark_learned, mayseen_benchmark_unlearned, mayseen_test_learned, \
        mayseen_test_unlearned, mayseen_union_learned, mayseen_union_unlearned = read_data(experiment)

        fig, axes = plt.subplots(2, 2, figsize=(8, 4), )  # sharex=True, sharey=True

        ax1 = axes[0, 0]
        ax1.hist(seen_union_learned, bins=70, color='blue', density=True, alpha=0.2, )
        ax1.hist(unseen_union_learned, bins=70, color='red', density=True, alpha=0.2, )
        # ax1.hist(mayseen_union_learned, bins=70, color='green', density=True, alpha=0.2, )
        ax1.set_title('test samples')
        ax1.set_xlabel('Loss Gap')
        ax1.set_ylabel('Probability Density')
        ax1.patch.set_facecolor('lightgray')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = axes[0, 1]
        ax2.hist(seen_union_unlearned, bins=70, color='blue', density=True, alpha=0.2, )
        ax2.hist(unseen_union_unlearned, bins=70, color='red', density=True, alpha=0.2, )
        # ax2.hist(mayseen_union_unlearned, bins=70, color='green', density=True, alpha=0.2, )
        ax2.set_title('samples labeled unlearn')
        ax2.set_xlabel('Loss Gap')
        ax2.set_ylabel('Probability Density')
        ax2.patch.set_facecolor('lightgray')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3 = axes[1, 0]
        ax3.hist(seen_test_learned, bins=70, color='blue', density=True, alpha=0.5, )
        ax3.hist(unseen_test_learned, bins=70, color='red', density=True, alpha=0.5, )
        ax3.hist(mayseen_test_learned, bins=70, color='green', density=True, alpha=0.5, )
        ax3.set_xlabel('Loss Gap')
        ax3.set_ylabel('Probability Density')
        ax3.patch.set_facecolor('lightgray')
        ax3.legend(['test samples', 'samples labeled unlearn'])
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax4 = axes[1, 1]
        ax4.hist(seen_test_unlearned, bins=70, color='blue', edgecolor='black', density=True, alpha=0.3, )
        ax4.hist(unseen_test_unlearned, bins=70, color='red', density=True, alpha=0.3, )
        ax4.hist(mayseen_test_unlearned, bins=70, color='green', density=True, alpha=0.3, )

        ax4.set_xlabel('Loss Gap')
        ax4.set_ylabel('Probability Density')
        ax4.patch.set_facecolor('lightgray')
        ax4.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    else:
        pass
        # #distance = wasserstein_distance(union_unlearned, test_unlearned)
        # print(distance)
        # input_dim = 10000
        # latent_dim = 100  # 潜在空间的维度
        # num_samples = 1  # 生成样本的数量
        #
        #
        # class Generator(nn.Module):
        #     def __init__(self, latent_dim, input_dim):
        #         super(Generator, self).__init__()
        #         self.fc1 = nn.Linear(latent_dim + 1, 128)  # 添加一个额外的输入维度用于距离值
        #         self.fc2 = nn.Linear(128, input_dim)
        #
        #     def forward(self, z, d):
        #         x = torch.cat((z, d), dim=1)  # 将潜在向量和距离值连接起来
        #         x = torch.relu(self.fc1(x))
        #         x = torch.sigmoid(self.fc2(x))
        #         return x
        #
        #
        # # 创建生成器实例
        # generator = Generator(latent_dim, input_dim)
        #
        # # 定义优化器
        # optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #
        # # 生成一个潜在向量，用距离值作为条件
        # latent_vector = torch.rand(1, latent_dim)
        # condition = torch.tensor([[distance]], dtype=torch.float32)  # 使用距离作为条件
        #
        # # 使用生成器生成新的分布样本
        # new_distribution = generator(latent_vector, condition)
        # # 在这里，new_distribution 就是根据输入的距离和已知分布生成的新分布
        # # 如果需要多个样本，可以多次生成
        # latent_vectors = torch.rand(num_samples, latent_dim)
        # conditions = torch.tensor([[distance]] * num_samples, dtype=torch.float32)
        # new_distributions = generator(latent_vectors, conditions)
        #
        # experimental_data = np.array(new_distributions[0].tolist())
        #
        # ####正态分布
        # # conf = []
        # # for item in test_learned:
        # #     test_data = item
        # #     # 计算测试数据的核密度估计值
        # #
        # #     # 计算测试数据的概率密度（置信度）
        # #     confidence = np.mean(experimental_data > test_data)
        # #     conf.append(confidence)
        # #     print("置信度:", confidence)
        # #############################################
        #
        # ###核函数
        # import numpy as np
        # from sklearn.neighbors import KernelDensity
        #
        # # 示例的实验数据
        #
        # # 创建核密度估计模型
        # kde = KernelDensity(bandwidth=0.47)  # 可以根据需要调整带宽参数
        #
        # # 用实验数据训练模型
        # kde.fit(experimental_data.reshape(-1, 1))
        #
        # # 假设测试数据是一个常数
        # # 需要将测试数据转换为二维数组形式
        #
        # conf = []
        #
        # for item in test_learned:
        #     test_data = np.array([[item]])
        #     # 计算测试数据的核密度估计值
        #     log_density = kde.score_samples(test_data)
        #
        #     # 将对数概率密度转换为概率密度
        #     confidence = np.exp(log_density)
        #     conf.append(confidence[0])
        #     # print("置信度:", confidence)
        #
        # ###########################################
        #
        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure(figsize=(5, 4))
        #
        # plt.hist(conf, bins=50, color='red', density=True, alpha=0.5, )
        # plt.xlabel('benchmark')
        # plt.show()
