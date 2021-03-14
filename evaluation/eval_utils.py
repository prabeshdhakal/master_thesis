"""Helper functions for evaluation of model performance."""

# Partially based on the following works:
# (1) Charles R. Qi (https://github.com/charlesq34/frustum-pointnets) 
#       The main author of the Frustum PointNet paper. The source code was shared with Apache Licence v2.0.

import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.style as style


# Plot parameters
plt.rcParams["figure.figsize"] = (12,8) # change this as necessary
plt.rcParams.update({'font.size': 15}) # change this as necessary
style.use("seaborn-deep")


def parse_log(log_file, metric):
    """
    Function to parse the log file created during model training.
    
    Parameters:
        log_file (str): file path of the training log file
        metric (str): metric for which the values are to be extracted

    """

    train_metric = []
    val_metric = []

    temp_list = []

    for item in log_file:
        if metric == "total_loss":
            if item[:10] == "total_loss":
                temp_list.append(np.float(item.split(":")[1]))
        
        elif metric == "iou2d":
            if item[:5] == "iou2d":
                temp_list.append(np.float(item.split(":")[1]))
        
        elif metric == "iou3d":
            if item[:5] == "iou3d" and (item[:9] != "iou3d_0.7") and (item[:9] != "iou3d_0.5"):
                temp_list.append(np.float(item.split(":")[1]))            
        elif metric == "iou3d_0.5": # iou3d_0.5
            if item[:9] == "iou3d_0.5":
                temp_list.append(np.float(item.split(":")[1]))

    for idx, item in enumerate(temp_list):
        if idx % 2 == 0:
            train_metric.append(item)
        else:
            val_metric.append(item)
    
    return train_metric, val_metric


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_ap(TP, FP, data_len, use_07_metric=False):

    tp_cumsum = np.cumsum(TP)
    fp_cumsum = np.cumsum(FP)

    acc = np.sum(TP) / data_len

    rec = tp_cumsum / float(data_len)
    prec = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = voc_ap(rec, prec, use_07_metric)

    return acc, prec, rec, ap


def get_bar_plots_kitti(sn, rn, tn, plot_title, ap=True, ylim=None, save_path=None):
    
    """
    Create bar plots that allows for a comparison of the prediction
    performance of the Source Network, the Reference Networks, and the Transfer Networks
    trained in the study.

    Parameters:
        sn (float): performance figure of the source network
        rn (list): performance figures of the reference networks
        tn (list): performance figures of the transfer networks
        plot_title (str): the title to be given to the plot
        ap (bool): set y-axis label to "Average Precision" (default=True). If False, y-axis label is set to "Accuracy"
        ylim (float): (optional) set a higher value for the y-axis limit (default=None).
        save_path (str): (optional) filepath to save the figure in

    Returns: a matplotlib figure
    """

    fig, ax = plt.subplots()

    # x-axis label and ticks
    val_sizes = ["310", "620", "1552", "3104", "4656", "6208"]
    x = np.arange(len(val_sizes))

    # plot width
    width = 0.35

    # plot each bars (RN and TN) and the source network perf (line)
    rects1 = ax.bar(x - width/2, rn, width, label='Reference Networks', color="xkcd:wine red", edgecolor="xkcd:wine red")
    rects2 = ax.bar(x + width/2, tn, width, label='Transfer Networks', color="lightcoral", edgecolor="xkcd:wine red", alpha=0.2)
    line_sn = ax.axhline(y=sn, color="firebrick", linewidth=2, label="KITTI Source Network")

    ax.set_title(plot_title)
    if ap:
        ax.set_ylabel("Average Precision")
    else:
        ax.set_ylabel("Accuracy")
    ax.set_xlabel("Training set size of the Reference Network and the Transfer Network")

    if ylim:
        ax.set_ylim([0, ylim])

    ax.set_xticks(x)
    ax.set_xticklabels(val_sizes)
    ax.legend(title="Legend", bbox_to_anchor=(1.005, 1), loc="upper left")

    if save_path:
        plt.savefig("graphs/" + save_path, bbox_inches="tight")

    plt.show()


def get_bar_plots_sunrgbd(sn, rn, tn, plot_title, ap=True, ylim=None, save_path=None):

    """
    Create bar plots that allows for a comparison of the prediction
    performance of the Source Network, the Reference Networks, and the Transfer Networks
    trained in the study.

    Parameters:
        sn (float): performance figure of the source network
        rn (list): performance figures of the reference networks
        tn (list): performance figures of the transfer networks
        plot_title (str): the title to be given to the plot
        ap (bool): set y-axis label to "Average Precision" (default=True). If False, y-axis label is set to "Accuracy"
        ylim (float): (optional) set a higher value for the y-axis limit (default=None).
        save_path (str): (optional) filepath to save the figure in

    Returns: a matplotlib figure
    """

    fig, ax = plt.subplots()

    # x-axis label and ticks
    val_sizes = ["456", "912", "2280", "4560", "6840", "9120"]
    x = np.arange(len(val_sizes))

    # plot width
    width = 0.35

    # plot each bars (RN and TN) and the source network perf (line)
    rects1 = ax.bar(x - width/2, rn, width, label='Reference Networks', color="xkcd:wine red", edgecolor="darkred")
    rects2 = ax.bar(x + width/2, tn, width, label='Transfer Networks', color="lightcoral", edgecolor="darkred", alpha=0.2)
    line_sn = ax.axhline(y=sn, color="firebrick", linewidth=2, label="SUN RGB-D Source Network")

    ax.set_title(plot_title)
    if ap:
        ax.set_ylabel("Average Precision")
    else:
        ax.set_ylabel("Accuracy")
    ax.set_xlabel("Training set size of the Reference Network and the Transfer Network")

    if ylim:
        ax.set_ylim([0, ylim])

    ax.set_xticks(x)
    ax.set_xticklabels(val_sizes)
    ax.legend(title="Legend", bbox_to_anchor=(1.005, 1),loc="upper left")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_model_loss(train_metric, val_metric, plot_title, save_path):

    """
    Create line charts that allows for a comparison of the the model loss on the 
    training set and the validation set during the training.

    Parameters:
        train_metric (list): model loss on the training set
        val_metric (list): model loss on the test set
        plot_title (str): the title to be given to the plot
        save_path (str): (optional) filepath to save the figure in

    Returns: a matplotlib figure
    """
    
    epochs = len(train_metric)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_title, fontsize=19)

    ax1.plot(epochs, train_metric, linewidth=2, color="xkcd:wine red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.title.set_text("Training Set")

    ax2.plot(epochs, val_metric, linewidth=2, color="xkcd:wine red")
    ax2.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.title.set_text("Validation Set")

    plt.subplots_adjust(wspace=0.075)

    if save_path:
        plt.savefig( save_path, bbox_inches="tight")

    plt.show()

def plot_precision_recall_curve(
    acc_25, prec_25, rec_25, ap_25,
    acc_50, prec_50, rec_50, ap_50,
    acc_70, prec_70, rec_70, ap_70,
    plot_title):
    

    """
    Plot the Precision-Recall curves at 25%, 50% and 70% IOU thresholds.

    Returns: a matplotlib figure
    """

    prec_25_ = sorted(list(prec_25), reverse=True)
    prec_50_ = sorted(list(prec_50), reverse=True)
    prec_70_ = sorted(list(prec_70), reverse=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(plot_title)

    ax1.plot(rec_25, prec_25_, '-')
    ax1.set_ylabel('precision')
    ax1.set_xlabel("recall")
    ax1.title.set_text("IoU Threshold 0.25\nacc={}, AP={}".format(round(acc_25, 4), round(ap_25, 4)))

    ax2.plot(rec_50, prec_50_, '.-')
    ax2.set_xlabel('recall')
    ax2.set_ylabel('precision')
    ax2.title.set_text("IoU Threshold 0.50\nacc={}, AP={}".format(round(acc_50, 4), round(ap_50, 4)))

    ax3.plot(rec_70, prec_70_, '.-')
    ax3.set_xlabel('recall')
    ax3.set_ylabel('precision')
    ax3.title.set_text("IoU Threshold 0.70\nacc={}, AP={}".format(round(acc_70, 4), round(ap_70, 4)))

    plt.show()