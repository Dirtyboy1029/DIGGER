# -*- coding: utf-8 -*- 
# @Time : 2023/9/26 21:13 
# @Author : DirtyBoy 
# @File : correction.py
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance
from Digger.utils import remove_outliers
from Digger.experiments.merge_dis2distribution_plot import merge2distribution
from Digger.experiments.distribution_diff_plot import read_data
from Digger.utils import txt_to_list, metrics, save2pkl, read_pkl
from Digger.experiments.acc_f1 import evaluate
'''
mayseen_learned  unlearned:[1,2,4,6,10,11,12]  learned:[0,3,5,7,8,9,13]
'''


def fpr_thresholds(y_label, y_pre, target_fpr=0.1):
    fpr, tpr, thresholds = roc_curve(y_label, y_pre, pos_label=1)
    # print('fpr  tpr  thersholds')
    # for i, value in enumerate(thersholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    print('AUC score:', roc_auc)
    print(f"Best Threshold: {best_threshold}")
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(fpr, tpr, 'k-', label='ROC (area = {0:.3f})'.format(roc_auc), lw=2, color='black', alpha=0.6)

    target_fpr = 0.40
    color = 'blue'
    closest_index = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[closest_index]
    threshold_at_target_fpr = thresholds[closest_index]
    print(tpr_at_target_fpr)
    ax1.scatter(fpr[closest_index], tpr[closest_index], c=color, s=50,
                label=f'FPR = {fpr[closest_index]:.2f}   Threshold = {threshold_at_target_fpr:.3f}')
    plt.plot([fpr[closest_index], fpr[closest_index]], [0, tpr_at_target_fpr], color=color, linestyle='--')
    plt.plot([0, fpr[closest_index]], [tpr_at_target_fpr, tpr_at_target_fpr], color=color, linestyle='--')

    # plt.annotate(f'FPR = {fpr[closest_index]:.2f}', (fpr[closest_index], 0.02), color=color)
    plt.annotate(f'TPR = {tpr_at_target_fpr:.3f}', (0.02, tpr_at_target_fpr + 0.02), color=color, fontweight='bold')

    target_fpr = 0.35
    color = 'red'
    closest_index = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[closest_index]
    threshold_at_target_fpr = thresholds[closest_index]
    print(tpr_at_target_fpr)
    ax1.scatter(fpr[closest_index], tpr[closest_index], c=color, s=50,
                label=f'FPR = {fpr[closest_index]:.2f}   Threshold = {threshold_at_target_fpr:.3f}')
    plt.plot([fpr[closest_index], fpr[closest_index]], [0, tpr_at_target_fpr], color=color, linestyle='--')
    plt.plot([0, fpr[closest_index]], [tpr_at_target_fpr, tpr_at_target_fpr], color=color, linestyle='--')

    # plt.annotate(f'FPR = {fpr[closest_index]:.2f}', (fpr[closest_index], 0.02), color=color)
    plt.annotate(f'TPR = {tpr_at_target_fpr:.3f}', (0.02, tpr_at_target_fpr + 0.02), color=color, fontweight='bold')

    target_fpr = 0.30
    color = 'green'
    closest_index = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[closest_index]
    threshold_at_target_fpr = thresholds[closest_index]
    print(tpr_at_target_fpr)
    ax1.scatter(fpr[closest_index], tpr[closest_index], c=color, s=50,
                label=f'FPR = {fpr[closest_index]:.2f}   Threshold = {threshold_at_target_fpr:.3f}')
    ##label=f'Threshold at FPR {target_fpr:.2f} = {threshold_at_target_fpr:.3f}\nTPR = {tpr_at_target_fpr:.3f}'
    plt.plot([fpr[closest_index], fpr[closest_index]], [0, tpr_at_target_fpr], color=color, linestyle='--')
    plt.plot([0, fpr[closest_index]], [tpr_at_target_fpr, tpr_at_target_fpr], color=color, linestyle='--')

    # plt.annotate(f'FPR = {fpr[closest_index]:.2f}', (fpr[closest_index], 0.02), color=color)
    plt.annotate(f'TPR = {tpr_at_target_fpr:.3f}', (0.02, tpr_at_target_fpr + 0.02), color=color, fontweight='bold')

    target_fpr = 0.25
    color = 'purple'
    closest_index = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[closest_index]
    threshold_at_target_fpr = thresholds[closest_index]
    print(tpr_at_target_fpr)
    ax1.scatter(fpr[closest_index], tpr[closest_index], c=color, s=50,
                label=f'FPR = {fpr[closest_index]:.2f}   Threshold = {threshold_at_target_fpr:.3f}')
    ##label=f'Threshold at FPR {target_fpr:.2f} = {threshold_at_target_fpr:.3f}\nTPR = {tpr_at_target_fpr:.3f}'
    plt.plot([fpr[closest_index], fpr[closest_index]], [0, tpr_at_target_fpr], color=color, linestyle='--')
    plt.plot([0, fpr[closest_index]], [tpr_at_target_fpr, tpr_at_target_fpr], color=color, linestyle='--')

    # plt.annotate(f'FPR = {fpr[closest_index]:.2f}', (fpr[closest_index], 0.02), color=color)
    plt.annotate(f'TPR = {tpr_at_target_fpr:.3f}', (0.02, tpr_at_target_fpr + 0.02), color=color, fontweight='bold')

    ax1.set_title('Comparison of ROC Curves for learned and Unlearned Samples', fontweight='bold', fontsize=14)
    ax1.set_xlabel("False Positive Rate", fontweight='bold', fontsize=14)
    ax1.set_ylabel("True Positive Rate", fontweight='bold', fontsize=14)

    legend = ax1.legend(loc="lower right")
    for text in legend.get_texts():
        text.set_weight('bold')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    ax1.patch.set_facecolor('lightgray')
    ax1.grid(True, linestyle='--', alpha=0.6)
    #plt.savefig('Figure/roc.pdf')
    plt.show()


def get_mayseen_label_and_data():
    unlearned_book_num = [1, 2, 4, 6, 10, 11, 12]
    learned_book_num = [0, 3, 5, 7, 8, 9, 13]
    '''
    unlearned:0
    learned:1
    
    '''
    vanilla_test_learned = read_pkl('intermediate_file/mayseen/vanilla/test_learned.pkl')
    vanilla_test_learned = vanilla_test_learned.reshape((len(vanilla_test_learned) * 200, -1)).squeeze()
    target_test_learned = read_pkl('intermediate_file/mayseen/target/test_learned.pkl')
    target_test_learned = target_test_learned.reshape((len(target_test_learned) * 200, -1)).squeeze()

    gt_labels = np.zeros(2800, dtype=int)
    learned_num = []
    unlearned_num = []
    for i in unlearned_book_num:
        unlearned_num = unlearned_num + list(range(i * 200, (i + 1) * 200))
    for i in learned_book_num:
        learned_num = learned_num + list(range(i * 200, (i + 1) * 200))
        gt_labels[i * 200:(i + 1) * 200] = 1

    return list(gt_labels), target_test_learned[learned_num], vanilla_test_learned[unlearned_num]


def corr_with_kde(new_bar, test_data):
    data = np.array(new_bar).reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.75, kernel='gaussian')
    kde.fit(data)
    print('find Maximum value of the probability density function......')

    x_test = np.linspace(data.min(), data.max(), 10000)

    # 计算估计的概率密度
    test_log_density = kde.score_samples(x_test.reshape(-1, 1))
    test_density = np.exp(test_log_density)

    # 找到概率密度的最大值
    max_density_index = np.argmax(test_density)
    center_value = x_test[max_density_index]
    print("Maximum value is " + str(center_value))
    test_learned = np.array(test_data).reshape(-1, 1)
    plt.hist(test_learned, bins=70, color='blue', density=True, alpha=0.2, )
    # ax7.hist(seen_test_unlearned, bins=70, color='red', density=True, alpha=0.2, )
    plt.hist(new_bar_learned, bins=70, color='green', density=True, alpha=0.5, )
    plt.show()
    log_densities = kde.score_samples(test_learned)
    density = np.exp(log_densities)
    for i, item in enumerate(test_learned):
        if item >= center_value:
            print(item, '    100%')
        else:
            print(item, '  ', density[i])


def find_percent_thr(list, precent):
    list1 = list
    list1.sort()
    percentile_index = int(len(list1) * precent / 100)
    height_threshold = list1[percentile_index]
    return height_threshold


if __name__ == '__main__':
    print('loading data...')
    experiment_type = 'mayseen'
    correction_type = 'normal'
    benchmark_learned, benchmark_unlearned, test_learned, test_unlearned, union_learned, union_unlearned = read_data(
        experiment_type)
    if experiment_type == 'mayseen':
        dis = -wasserstein_distance(union_unlearned, test_unlearned)
        mayseen_gt_labels, learned_samples, unlearned_samples = get_mayseen_label_and_data()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.boxplot([learned_samples, unlearned_samples], labels=['learned', 'unlearned'], showfliers=False)
        plt.title("Male Height Box Plot")
        plt.xlabel("Height (cm)")

        plt.tight_layout()
        plt.show()
    else:
        if experiment_type == 'seen':
            dis = -wasserstein_distance(union_unlearned, test_unlearned)
        else:
            dis = wasserstein_distance(union_unlearned, test_unlearned)
        mayseen_gt_labels = None
        learned_samples = None
        unlearned_samples = None

    print('load ' + experiment_type + ' data finish!!!!')
    print('Distance between distributions is ' + str(dis))
    new_bar_learned = merge2distribution(dis, benchmark_learned)
    if correction_type == 'kde':
        corr_with_kde(new_bar_learned, test_learned)
    else:

        bar_mean = np.mean(new_bar_learned)
        bar_std = np.std(new_bar_learned)
        conf_learned = []
        conf_unlearned = []
        conf = []
        for i, item in enumerate(test_learned):
            probability = norm.cdf(item, loc=bar_mean, scale=bar_std)
            if i < 1400:
                conf_learned.append(probability)
            else:
                conf_unlearned.append(probability)
            conf.append(probability)

        if experiment_type == 'mayseen':
            #thr = 0.254305
            #fpr_thresholds([1] * 1400 + [0] * 1400, conf)
            evaluate(np.array(conf),np.array([1] * 1400 + [0] * 1400), threshold=0.10295975837589744)
            # plt.hist(conf_unlearned, bins=100, color='red', density=False, alpha=0.2, )
            # plt.hist(conf_learned, bins=100, color='blue', density=False, alpha=0.2, )
        elif experiment_type == 'seen':
            #plt.hist(conf, bins=100, color='blue', density=False, alpha=0.2, )
            num_ = np.zeros(10)
            for item in conf:
                if int(item/0.1)>9:
                    num_[9] = num_[9]+1
                else:
                    num_[int(item/0.1)] = num_[int(item/0.1)] + 1
            print(num_)
        else:
            num_ = np.zeros(10)
            for item in conf:
                if int(item / 0.1) > 9:
                    num_[9] = num_[9] + 1
                else:
                    num_[int(item / 0.1)] = num_[int(item / 0.1)] + 1
            for itme in num_:
                print(int(itme))
        plt.show()

'''
0.05729774961397119
0.06724043850928503
0.07972133994327779
0.10295975837589744


0.9757142857142858
0.9657142857142857
0.955
0.9242857142857143


'''