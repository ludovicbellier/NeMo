"""
NeMo: Neural Modeling toolbox.
A series of Python tools and algorithms for applying predictive models to neural activity.
"""

import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


########################################################################################################################
#  Data preparation
########################################################################################################################
def create_classes(s, vocal_boundaries, fs, flag_fig=0, **kwargs):
    voc = np.asarray(vocal_boundaries) * fs
    dim = s.shape
    c = np.zeros(dim[0])
    if len(voc) > 0:
        c[voc[0]:voc[1]] = 1
    if flag_fig > 0:
        plt.figure(figsize=(c.shape[0] / 800, 4))
        plt.imshow(10 * np.log10(s.T), aspect='auto', origin='lower', cmap='jet', clim=(-10, 10))
        plt.plot(c * (dim[1] - 1), color='r', linewidth=4)
        plt.title('classes')
    return c


def create_groups(c, group_length, fs, fixed_test, **kwargs):
    n_groups = int(np.floor(c.shape[0] / (group_length * fs)))
    g = np.repeat(range(n_groups), group_length * fs)
    g = np.concatenate((g, np.repeat(g[-1], c.shape[0] - len(g))))
    idx = 0
    while idx <= n_groups:
        g_idx = np.where(g == idx)[0]
        g_s = c[g_idx]
        if len(np.unique(g_s)) > 1:
            g[g_idx[g_s == g_s[0]]] -= 1
            g[g > idx] -= 1
            n_groups = np.max(g)
        else:
            idx += 1
    if len(fixed_test) > 0:
        idx_fixed_test = np.multiply(fixed_test, fs).astype('int16')
        idx_fixed_test = np.arange(idx_fixed_test[0], idx_fixed_test[1])
        idx_fixed_test = idx_fixed_test[idx_fixed_test < len(g)]
        g_fixed_test = np.unique(g[idx_fixed_test])
        idx = 0
        while idx < len(g_fixed_test):
            g_idx = np.where(g == g_fixed_test[idx])[0]
            if len(np.unique(np.in1d(g_idx, idx_fixed_test))) > 1:
                idx_in_group = np.in1d(g_idx, idx_fixed_test)  # type: np.bool
                if idx_in_group[0]:
                    g[g_idx[~idx_in_group]] += 1
                else:
                    g[g_idx[~idx_in_group]] -= 1
            else:
                idx += 1
    return g


def fix_artifacts(x, y, a, c, method, model_type, flag_fig=0, **kwargs):
    if flag_fig > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(211)
        if model_type == 'encoding':
            plt.plot(y)
            plt.plot(np.where(a)[0], y[np.where(a)[0]], '.r')
        else:
            plt.imshow(x.T, aspect='auto', origin='lower', clim=(-.5, 1))
            y_ = np.zeros(x.shape[0])
            plt.plot(np.where(a)[0], y_[np.where(a)[0]], '.r')
    idx = np.array(a, dtype=bool)
    if method == 'remove':
        if len(idx.shape) > 1:
            idx = np.where(idx.any(1))[0]
        else:
            idx = np.where(idx)[0]
        [x, y, c, a] = [np.delete(array, idx, axis=0) for array in [x, y, c, a]]
    elif method == 'replace':
        for idxChan in range(x.shape[1]):
            if np.any(idx[:, idxChan]):
                x[idx[:, idxChan], idxChan] = np.median(x[~idx[:, idxChan], idxChan])
    if flag_fig > 0:
        plt.subplot(212)
        if model_type == 'encoding':
            plt.plot(y)
        else:
            plt.imshow(x.T, aspect='auto', origin='lower', clim=(-.5, 1))
    return x, y, a, c


def build_lag_matrix(x, y, a, c, lags, offset, fs, model_type, **kwargs):
    lags = int(lags * fs)
    offset = int(offset * fs)
    length = x.shape[0] - lags + 1
    x_lag_mat = np.zeros((length, x.shape[1] * lags))
    for i in range(length):
        x_lag_mat[i, :] = np.reshape(x[i:i + lags, :], (1, x.shape[1] * lags))
    if model_type == 'encoding':
        idx_y = lags - 1 - offset
        if idx_y >= 0:
            [y, a] = [M[idx_y:] for M in [y, a]]
            c = c[:length]
        else:
            x_lag_mat = x_lag_mat[-idx_y:]
            c = c[-idx_y:-lags + 1]
        if y.shape[0] > x_lag_mat.shape[0]:
            [y, a] = [M[:-offset] for M in [y, a]]
    else:
        if offset >= 0:
            [y, a, c] = [M[offset:-lags + offset + 1] for M in [y, a, c]]
        else:
            x_lag_mat = x_lag_mat[offset:]
            if y.shape[0] > x_lag_mat.shape[0]:
                [y, a, c] = [M[:x_lag_mat.shape[0]] for M in [y, a, c]]
    if len(y.shape) == 1:
        idx_ok_y = ~np.isnan(y)
    else:
        idx_ok_y = ~np.isnan(y).any(axis=1)
    x_lag_mat = x_lag_mat[idx_ok_y, :]
    [y, a, c] = [M[idx_ok_y] for M in [y, a, c]]
    idx_ok_x = ~np.isnan(x_lag_mat).any(axis=1)
    [y, a, c] = [M[idx_ok_x] for M in [y, a, c]]
    x_lag_mat = x_lag_mat[idx_ok_x]
    if len(y.shape) == 1:
        y = np.reshape(y, [x_lag_mat.shape[0], 1])
    return x_lag_mat, y, a, c


def get_feat_corr(x, lag_span, step, fs):
    lag_span_samp = int(lag_span * fs)
    steps = list(range(0, x.shape[0] - lag_span_samp, int(step * fs)))
    c = np.zeros((len(steps), lag_span_samp))
    for i in enumerate(steps):
        if i[1] > 0 and i[1] % 100 == 0:
            print('{}/{}'.format(i[1], steps[-1]))
        c[i[0], :] = [np.corrcoef(x[i[1], :], x[i[1] + j, :])[0, 1] for j in range(lag_span_samp)]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(3, 1, (1, 2))
    plt.imshow(c, aspect='auto', extent=(0, lag_span - 1 / fs, steps[-1] / fs, 0))
    plt.ylabel('time of i (s)')
    ax.set_xticklabels([])
    fig.add_subplot(313)
    plt.hlines(.5, 0, lag_span, colors=(.7, .7, .7))
    plt.plot(np.arange(0, lag_span, 1 / fs), np.nanmean(c, 0))
    plt.xlim((0, lag_span - 1 / fs))
    plt.xlabel('lag of j from i (s)')
    plt.ylabel('correlation coefficient (i, j)')
    plt.title('mean of {} estimates across dataset'.format(len(steps)))


########################################################################################################################
#  Data splitting and scaling
########################################################################################################################
def stratified_group_shuffle_split(g, c, split_ratio, fixed_test, fs, flag_fig=0, **kwargs):
    groups = np.unique(g)
    n_groups = len(groups)
    n_classes = len(np.unique(c))
    n_sets = len(split_ratio)
    n_groups_per_set = np.zeros((n_classes, n_sets))  # n_classes * n_sets
    # classes info
    if n_classes == 1 and np.unique(c)[0] == 1:
        c = c - np.unique(c)[0]
    g_classes = np.asarray([np.round(c[g == i].mean()) for i in range(n_groups)])
    n_groups_per_class = np.asarray([np.sum(g_classes == i) for i in range(n_classes)])
    # test set
    if len(fixed_test) > 0:  # map fixed test set to groups
        fixed_test_idx = np.multiply(fixed_test, fs).astype('int16')
        g_test = g[fixed_test_idx[0]:fixed_test_idx[1]]
        group_test = np.unique(g_test)
        groups = np.setdiff1d(groups, group_test)
        idx = g_classes[group_test]
        g_classes = np.delete(g_classes, group_test)
        n_groups_per_set[:, -1] = [sum(idx == i) for i in range(n_classes)]
    else:
        n_groups_per_set[:, -1] = np.round(split_ratio[-1] * n_groups_per_class).astype('int')
        group_test = np.array(())
    # validation set (optional)
    group_val = np.array(())
    if n_sets > 2:
        n_groups_per_set[:, 1] = np.round(split_ratio[1] * n_groups_per_class).astype('int')
    # training set
    group_train = np.array(())
    n_groups_per_set[:, 0] = n_groups_per_class - n_groups_per_set[:, 1:].sum(1)
    n_groups_per_set = n_groups_per_set.astype('int')
    for s in np.unique(c).astype('int'):
        group_tmp = groups[g_classes == s]
        np.random.shuffle(group_tmp)
        group_train = np.append(group_train, group_tmp[:n_groups_per_set[s, 0]])
        if n_sets > 2:
            group_val = np.append(group_val, group_tmp[n_groups_per_set[s, 0]:np.sum(n_groups_per_set[s, :-1])])
        if len(fixed_test) == 0:
            group_test = np.append(group_test, group_tmp[np.sum(n_groups_per_set[s, :-1]):])
    if n_sets > 2:
        all_groups = [group_train, group_val, group_test]
    else:
        all_groups = [group_train, group_test]
    split_indices = [np.where(np.in1d(g, gid))[0] for gid in all_groups]
    if flag_fig > 0:
        line_style = ''
        marker = '.'
        marker_size = 1
        plt.figure(figsize=(10, 5))
        colors_list = [(.2, .5, 1), 'tab:orange', 'r']
        tb = np.arange(0, 188.48, .01)
        for i in range(n_sets):
            plt.plot(tb[split_indices[i]], g[split_indices[i]], ls=line_style, marker=marker, ms=marker_size,
                     c=colors_list[i])
        if len(np.unique(c)) > 1:
            plt.plot(tb, c / (n_classes - 1) * (g[-1] + 2) - 1, c='black')
        if n_sets > 2:
            str_legend = ('training set', 'validation set', 'test set', 'classes')
        else:
            str_legend = ('training set', 'test set', 'classes')
        plt.legend(str_legend)
        plt.xlabel('time (s)')
        plt.ylabel('group index')
        plt.xlim((tb[0], tb[-1]))
    return split_indices


def get_n_splits(groups, classes, params, n_splits=5, **kwargs):
    split_indices = []
    for _ in range(n_splits):
        split_indices.append(stratified_group_shuffle_split(groups, classes, **params))
    return split_indices


def perform_split(x, y, split_indices, n_feat_y, flag_log, flag_fig=0, **kwargs):
    n_sets = len(split_indices)
    feature_sets = [x[split_indices[i]] for i in range(n_sets)]
    target_sets = [y[split_indices[i]] for i in range(n_sets)]
    if flag_fig > 0:
        c_min = np.percentile(x[:, :n_feat_y], 5)
        c_max = np.percentile(x[:, :n_feat_y], 95)
        clim = (c_min, c_max)
        clim2 = (0, 0)
        l_split = [len(ind) for ind in split_indices]
        l_split = np.divide(l_split, np.sum(l_split))
        l_split = np.cumsum(np.round(l_split * 10)).astype('int')
        grid = plt.GridSpec(2, l_split[-1])
        plt.figure(figsize=(12, 5))
        ax_tn_f = plt.subplot(grid[0, :l_split[0]])
        plt.imshow(feature_sets[0][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.title('training set')
        ax_tn_t = plt.subplot(grid[1, :l_split[0]], sharex=ax_tn_f)
        n_y = target_sets[-1].shape[1]
        if n_y > 1:
            if flag_log == 1:
                c_min2 = np.percentile(np.exp(target_sets[0])[:, :n_feat_y], 5)
                c_max2 = np.percentile(np.exp(target_sets[0])[:, :n_feat_y], 95)
                clim2 = (c_min2, c_max2)
                plt.imshow(np.exp(target_sets[0]).T, aspect='auto', origin='lower', clim=clim2)
            else:
                c_min2 = np.percentile(target_sets[0][:, :n_feat_y], 5)
                c_max2 = np.percentile(target_sets[0][:, :n_feat_y], 95)
                clim2 = (c_min2, c_max2)
                plt.imshow(target_sets[0].T, aspect='auto', origin='lower', clim=clim2)
        else:
            if flag_log == 1:
                plt.plot(np.exp(target_sets[0]))
            else:
                plt.plot(target_sets[0])
        plt.xlim(0, target_sets[0].shape[0])
        ax_vl_f = plt.subplot(grid[0, l_split[0]:l_split[1]])
        plt.imshow(feature_sets[1][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.gca().axes.yaxis.set_ticklabels([])
        if n_sets > 2:
            plt.title('validation set')
        else:
            plt.title('test set')
        ax_vl_t = plt.subplot(grid[1, l_split[0]:l_split[1]], sharex=ax_vl_f, sharey=ax_tn_t)
        if n_y > 1:
            if flag_log == 1:
                plt.imshow(np.exp(target_sets[1]).T, aspect='auto', origin='lower', clim=clim2)
            else:
                plt.imshow(target_sets[1].T, aspect='auto', origin='lower', clim=clim2)
        else:
            if flag_log == 1:
                plt.plot(np.exp(target_sets[1]))
            else:
                plt.plot(target_sets[1])
        plt.xlim(0, target_sets[1].shape[0])
        plt.setp(ax_vl_t.get_yticklabels(), visible=False)
        if n_sets > 2:
            ax_tt_f = plt.subplot(grid[0, l_split[1]:l_split[2]])
            plt.imshow(feature_sets[2][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.title('test set')
            ax_tt_t = plt.subplot(grid[1, l_split[1]:l_split[2]], sharex=ax_tt_f, sharey=ax_tn_t)
            if n_y > 1:
                if flag_log == 1:
                    plt.imshow(np.exp(target_sets[2]).T, aspect='auto', origin='lower', clim=clim2)
                else:
                    plt.imshow(target_sets[2].T, aspect='auto', origin='lower', clim=clim2)
            else:
                if flag_log == 1:
                    plt.plot(np.exp(target_sets[2]))
                else:
                    plt.plot(target_sets[2])
            plt.xlim(0, target_sets[2].shape[0])
            plt.setp(ax_tt_t.get_yticklabels(), visible=False)
    return feature_sets, target_sets


def scale(feature_sets, n_feat_y, scaler_type, scaler_iqr, flag_fig=0, **kwargs):
    n_sets = len(feature_sets)
    grid = []
    l_split = []
    ax_tn_1 = []
    ax_vl_1 = []
    ax_tt_1 = []
    fig = []
    if flag_fig > 0:
        l_split = [s.shape[0] for s in feature_sets]
        l_split = np.divide(l_split, np.sum(l_split))
        l_split = np.cumsum(np.round(l_split * 10)).astype('int')
        c_min = np.min([np.percentile(s, 5) for s in feature_sets])
        c_max = np.max([np.percentile(s, 95) for s in feature_sets])
        clim = (c_min, c_max)
        grid = plt.GridSpec(2, l_split[-1])
        fig = plt.figure(figsize=(12, 5))
        ax_tn_1 = plt.subplot(grid[0, :l_split[0]])
        plt.imshow(feature_sets[0][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.title('training set')
        ax_vl_1 = plt.subplot(grid[0, l_split[0]:l_split[1]])
        plt.imshow(feature_sets[1][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.gca().axes.yaxis.set_ticklabels([])
        if n_sets > 2:
            plt.title('validation set')
        else:
            plt.title('test set')
        if n_sets > 2:
            ax_tt_1 = plt.subplot(grid[0, l_split[1]:l_split[2]])
            im1 = plt.imshow(feature_sets[2][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.title('test set')
            fig.subplots_adjust(right=0.9)
            cbar_ax1 = fig.add_axes([0.915, .53, 0.02, 0.35])
            fig.colorbar(im1, cax=cbar_ax1)
    if scaler_type == 'robust':
        scaler = RobustScaler(quantile_range=scaler_iqr)
    elif scaler_type == 'minMax':
        scaler = MinMaxScaler(feature_range=scaler_iqr)
    else:
        scaler = StandardScaler()
    feature_sets[0] = scaler.fit_transform(feature_sets[0])
    for i in range(1, n_sets):
        feature_sets[i] = scaler.transform(feature_sets[i])
    if flag_fig > 0:
        c_min = np.min([np.percentile(s, 5) for s in feature_sets])
        c_max = np.max([np.percentile(s, 95) for s in feature_sets])
        clim = (c_min, c_max)
        plt.subplot(grid[1, :l_split[0]], sharex=ax_tn_1, sharey=ax_tn_1)
        plt.imshow(feature_sets[0][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.subplot(grid[1, l_split[0]:l_split[1]], sharex=ax_vl_1, sharey=ax_vl_1)
        plt.imshow(feature_sets[1][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
        plt.gca().axes.yaxis.set_ticklabels([])
        if n_sets > 2:
            plt.subplot(grid[1, l_split[1]:l_split[2]], sharex=ax_tt_1, sharey=ax_tt_1)
            im2 = plt.imshow(feature_sets[2][:, :n_feat_y].T, aspect='auto', origin='lower', clim=clim)
            plt.gca().axes.yaxis.set_ticklabels([])
            c_bar_ax2 = fig.add_axes([0.915, 0.11, 0.02, 0.35])
            fig.colorbar(im2, cax=c_bar_ax2)
    return feature_sets, scaler


########################################################################################################################
#  Core models
########################################################################################################################
class RobustMultipleLinearRegressionEarlyStopping(BaseEstimator, RegressorMixin):
    """ Custom estimator for robust multiple linear regression with (optional) early stopping """

    def __init__(self, learning_rate=.001, n_iter_no_change=10, tol=0, max_iter=200, early_stopping=False, **kwargs):
        self.learning_rate = learning_rate
        self.n_iter_no_change = n_iter_no_change
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping

        self.is_fitted_ = False
        self.best_coef_ = np.zeros([0, 1])
        self.best_intercept_ = np.array([0])
        self.best_loss_ = 0
        self.best_iteration_ = 0
        self.run_time_ = 0
        self.train_loss_history_ = np.empty(shape=[0])
        if self.early_stopping:
            self.val_loss_history_ = np.empty(shape=[0])

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        t = time.time()
        n_target = y_train.shape[1]
        if n_target == 1:
            x_train, y_train = check_X_y(x_train, y_train.ravel(), accept_sparse=True)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape((y_train.shape[0], 1))
            if self.early_stopping:
                y_val = y_val.reshape((y_val.shape[0], 1))
        n_feat = x_train.shape[1]
        n_target = y_train.shape[1]
        self.best_coef_ = np.zeros([n_feat, 1])
        tf.reset_default_graph()
        x_ = tf.placeholder(tf.float32, shape=(None, n_feat))
        y_ = tf.placeholder(tf.float32, shape=(None, n_target))
        coef_ = tf.Variable(tf.zeros((n_feat, n_target)))
        intercept_ = tf.Variable(tf.constant((y_train.mean(),)))
        predict_operation = tf.add(tf.matmul(x_, coef_), intercept_)
        loss = tf.losses.huber_loss(y_, predict_operation)
        optimize_operation = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        counter = 0
        iteration = 1
        val_loss = 0
        while counter < self.n_iter_no_change:
            _, train_loss = sess.run([optimize_operation, loss], feed_dict={x_: x_train, y_: y_train})
            self.train_loss_history_ = np.append(self.train_loss_history_, train_loss)
            if self.early_stopping:
                val_loss = sess.run(loss, feed_dict={x_: x_val, y_: y_val})
                self.val_loss_history_ = np.append(self.val_loss_history_, val_loss)
                loss_monitor = val_loss
            else:
                loss_monitor = train_loss
            if iteration == 1:
                self.best_loss_ = loss_monitor
            else:
                if loss_monitor >= self.best_loss_ - self.tol:
                    counter += 1
                else:
                    counter = 0
                    self.best_coef_ = sess.run(coef_)
                    self.best_intercept_ = sess.run(intercept_)
                    self.best_loss_ = loss_monitor
                    self.best_iteration_ = iteration
            if iteration % 20 == 1:
                if self.early_stopping:
                    str_val = 'val {:.3g} '.format(val_loss)
                else:
                    str_val = ''
                print('iter {:3d} - loss: train {:.3g} {}- count {:2d}'.format(iteration, train_loss, str_val, counter))
            iteration += 1
            if iteration == self.max_iter and self.n_iter_no_change < self.max_iter:
                self.best_iteration_ = self.max_iter
                print("WARNING: iteration maximum limit has been reached -> breaking...")
                break
            if np.isnan(loss_monitor):
                break
        self.is_fitted_ = True
        self.run_time_ = time.time() - t
        return self

    def predict(self, x):
        x = check_array(x, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        y_predicted = np.matmul(x, self.best_coef_) + self.best_intercept_
        return y_predicted


class RegularizedRegressionCustomEstimator(BaseEstimator, RegressorMixin):
    """ Custom estimator from Ridge, Lasso or ElasticNet """

    def __init__(self, alpha=1, tol=.001, max_iter=None, random_state=None, algo='ridge', l1_ratio=.5, **kwargs):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.algo = algo
        self.l1_ratio = l1_ratio
        self.is_fitted_ = False
        self.best_model_ = None
        self.best_coef_ = np.zeros([0, 1])
        self.best_loss_ = 0
        self.best_iteration_ = 0
        self.run_time_ = 0
        self.train_loss_history_ = np.empty(shape=[0])

    def fit(self, x, y):
        t = time.time()
        n_target = y.shape[1]
        if n_target == 1:
            x, y = check_X_y(x, y.ravel(), accept_sparse=True)
        if self.algo == 'ridge':
            mdl = Ridge(alpha=self.alpha, normalize=True, max_iter=self.max_iter,
                        tol=self.tol, random_state=self.random_state)
        elif self.algo == 'lasso':
            mdl = Lasso(alpha=self.alpha, random_state=self.random_state)
        else:
            mdl = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
        mdl.fit(x, y)
        self.best_model_ = deepcopy(mdl)
        self.best_coef_ = mdl.coef_
        if mdl.n_iter_ is None:
            self.best_iteration_ = 0
        else:
            self.best_iteration_ = mdl.n_iter_
        self.is_fitted_ = True
        self.run_time_ = time.time() - t
        return self

    def predict(self, x):
        x = check_array(x, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        y_predicted = self.best_model_.predict(x)
        return y_predicted


class MLPRegressorCustomEarlyStopping(BaseEstimator, RegressorMixin):
    """ Custom estimator from MLPRegressor with custom early stopping """

    def __init__(self, hidden_layer_sizes=(100,), alpha=.0001, batch_size='auto', learning_rate=.001,
                 max_iter=200, random_state=None, tol=.0001, early_stopping=False, n_iter_no_change=10,
                 split_ratio=(.6, .2, .2), **kwargs):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.split_ratio = split_ratio
        self.is_fitted_ = False
        self.best_model_ = None
        self.best_coef_ = np.zeros([0, 1])
        self.loss_monitor_ = 0
        self.best_loss_ = 0
        self.best_iteration_ = 0
        self.run_time_ = 0
        self.train_loss_history_ = np.empty(shape=[0])
        if self.early_stopping:
            self.val_loss_history_ = np.empty(shape=[0])

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        t = time.time()
        n_target = y_train.shape[1]
        if n_target == 1:
            x_train, y_train = check_X_y(x_train, y_train.ravel(), accept_sparse=True)
        if self.early_stopping:
            if self.early_stopping == 'built_in':
                mdl = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, max_iter=self.max_iter,
                                   learning_rate_init=self.learning_rate, n_iter_no_change=self.n_iter_no_change,
                                   batch_size=self.batch_size, random_state=self.random_state, early_stopping=True,
                                   validation_fraction=self.split_ratio[1] / np.sum(self.split_ratio[:2]))
                mdl.fit(x_train, y_train)
                self.train_loss_history_ = mdl.loss_curve_
                self.val_loss_history_ = mdl.validation_scores_
                self.best_loss_ = mdl.best_validation_score_
                self.best_model_ = deepcopy(mdl)
                self.best_iteration_ = mdl.n_iter_ - self.n_iter_no_change - 1
                self.best_coef_ = mdl.coefs_[0].mean(1)
            else:
                mdl = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, max_iter=1,
                                   learning_rate_init=self.learning_rate, n_iter_no_change=self.n_iter_no_change,
                                   batch_size=self.batch_size, random_state=self.random_state, early_stopping=False,
                                   warm_start=True)
                counter = 0
                iteration = 1
                while counter < self.n_iter_no_change:
                    mdl.fit(x_train, y_train)
                    y_pred_train = mdl.predict(x_train)
                    train_loss = mean_squared_error(y_train, y_pred_train)
                    self.train_loss_history_ = np.append(self.train_loss_history_, train_loss)
                    y_pred_val = mdl.predict(x_val)
                    val_loss = mean_squared_error(y_val, y_pred_val)
                    self.val_loss_history_ = np.append(self.val_loss_history_, val_loss)
                    loss_monitor = val_loss
                    if iteration == 1:
                        self.loss_monitor_ = loss_monitor
                        self.best_model_ = deepcopy(mdl)
                    else:
                        if loss_monitor >= self.loss_monitor_ - self.tol:
                            counter += 1
                        else:
                            counter = 0
                            self.loss_monitor_ = loss_monitor
                            self.best_model_ = deepcopy(mdl)
                            self.best_iteration_ = iteration
                    if iteration % 5 == 1:
                        print('iter {:3d} - loss: train {:.3g} val {:.3g} - count {:2d}'.format(iteration, train_loss,
                                                                                                val_loss, counter))
                    iteration += 1
                    if iteration == self.max_iter and self.n_iter_no_change < self.max_iter:
                        print("WARNING: iteration maximum limit has been reached -> breaking...")
                        break
                    if np.isnan(loss_monitor):
                        break
                self.best_coef_ = self.best_model_.coefs_[0].mean(1)
                self.best_loss_ = self.loss_monitor_
        else:
            mdl = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, max_iter=self.max_iter,
                               learning_rate_init=self.learning_rate, n_iter_no_change=self.n_iter_no_change,
                               batch_size=self.batch_size, random_state=self.random_state)
            mdl.fit(x_train, y_train)
            self.train_loss_history_ = mdl.loss_curve_
            self.best_loss_ = mdl.loss_
            self.best_model_ = deepcopy(mdl)
            self.best_iteration_ = mdl.n_iter_ - self.n_iter_no_change
            self.best_coef_ = mdl.coefs_[0].mean(1)
        self.is_fitted_ = True
        self.run_time_ = time.time() - t
        return self

    def predict(self, x):
        x = check_array(x, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        y_predicted = self.best_model_.predict(x)
        return y_predicted


########################################################################################################################
#  Model outputs
########################################################################################################################
def get_predicted_sets(model, feature_sets):
    n_sets = len(feature_sets)
    predicted_sets = []
    for idx in range(n_sets):
        predicted_sets.append(model.predict(feature_sets[idx]))
    return predicted_sets


def get_model_metrics(model, feature_sets, target_sets, model_type, flag_log, verbose):
    n_sets = len(feature_sets)
    predicted_sets = get_predicted_sets(model, feature_sets)
    if model_type in ['decoding', 'recon'] and flag_log == 1:
        target_sets_tmp = [np.exp(x) for x in target_sets]
        predicted_sets_tmp = [np.exp(x) for x in predicted_sets]
    else:
        target_sets_tmp = target_sets.copy()
        predicted_sets_tmp = predicted_sets.copy()
    metrics = np.zeros((n_sets, 3))
    n_y = target_sets_tmp[-1].shape[1]
    if n_y > 1:
        if predicted_sets_tmp[0].shape[1] == n_y:
            metrics[:, 0] = [np.mean([np.corrcoef(target_sets_tmp[j][:, i], predicted_sets_tmp[j][:, i])[0, 1]
                                      for i in range(n_y)]) for j in range(n_sets)]
            metrics[:, 1] = [np.mean([r2_score(y[:, i], y_pred[:, i]) for i in range(n_y)])
                             for y, y_pred in zip(target_sets_tmp, predicted_sets_tmp)]
            metrics[:, 2] = [np.mean([mean_squared_error(y[:, i], y_pred[:, i]) for i in range(n_y)])
                             for y, y_pred in zip(target_sets_tmp, predicted_sets_tmp)]
    else:
        metrics[:, 0] = [np.corrcoef(y, y_pred, rowvar=False)[0, 1]
                         for y, y_pred in zip(target_sets_tmp, predicted_sets_tmp)]
        metrics[:, 1] = [r2_score(y, y_pred) for y, y_pred in zip(target_sets_tmp, predicted_sets_tmp)]
        metrics[:, 2] = [mean_squared_error(y, y_pred) for y, y_pred in zip(target_sets_tmp, predicted_sets_tmp)]
    if verbose > 0:
        if n_sets == 2:
            print('Best val. loss at iteration #{:d} after {:.2f} sec\n'
                  'Train r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                  'Test  r={:.3g} - R2={:.3g} - MSE={:.3g}\n'.format(model.best_iteration_, model.run_time_,
                                                                     metrics[0, 0], metrics[0, 1], metrics[0, 2],
                                                                     metrics[1, 0], metrics[1, 1], metrics[1, 2]))
        else:
            print('Best val. loss at iteration #{:d} after {:.2f} sec\n'
                  'Train r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                  'Val   r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                  'Test  r={:.3g} - R2={:.3g} - MSE={:.3g}\n'.format(model.best_iteration_, model.run_time_,
                                                                     metrics[0, 0], metrics[0, 1], metrics[0, 2],
                                                                     metrics[1, 0], metrics[1, 1], metrics[1, 2],
                                                                     metrics[2, 0], metrics[2, 1], metrics[2, 2]))
    return metrics


def get_model_output(model, feature_sets, target_sets, scaler, lags, fs, offset, model_type, n_feat_y, algo,
                     learning_rate, n_iter_no_change, hidden_layer_sizes, alpha, l1_ratio, n_target, flag_log,
                     early_stopping, verbose=1, flag_fig=0, **kwargs):
    if n_target > 1:
        if algo == 'MLP':
            coefs = model.best_coef_
        else:
            coefs = model.best_coef_.mean(1)
        if n_feat_y >= 1 and coefs.shape[0] == int(lags * fs) * n_feat_y:
            coefs = np.reshape(coefs, (int(lags * fs), int(np.ceil(n_feat_y)))).T
        else:
            coefs = np.zeros((int(lags * fs), int(np.ceil(n_feat_y))))
    else:
        if n_feat_y >= 1 and model.best_coef_.ravel().shape[0] == int(lags * fs) * n_feat_y:
            coefs = np.reshape(model.best_coef_.ravel(), (int(lags * fs), int(np.ceil(n_feat_y)))).T
        else:
            coefs = np.zeros((int(lags * fs), int(np.ceil(n_feat_y))))
    metrics = get_model_metrics(model, feature_sets, target_sets, model_type, flag_log, verbose)
    predicted_sets = get_predicted_sets(model, feature_sets)
    if algo in ['rMLRwES', 'linReg']:
        str_params = 'learning rate={} - n_iter_no_change={}'.format(learning_rate, n_iter_no_change)
    elif algo in ['ridge', 'lasso']:
        str_params = 'alpha={}'.format(alpha)
    elif algo == 'elasticNet':
        str_params = 'alpha={} - l1_ratio={}'.format(alpha, l1_ratio)
    else:
        str_params = 'hidden_layer_sizes={} - alpha={}'.format(hidden_layer_sizes, alpha)
    if flag_fig > 0:
        if model_type == 'encoding':
            extent_tuple = (round((offset - lags) * 1000), round(offset * 1000), 1, n_feat_y + 1)
            coefs_y_label = 'frequency bin'
        else:
            extent_tuple = (-round(offset * 1000), round((lags - offset) * 1000), 1, n_feat_y + 1)
            coefs_y_label = 'electrode'

        plt.figure(figsize=(20, 10))
        if n_target > 1:
            grid = plt.GridSpec(3, 10)
        else:
            grid = plt.GridSpec(2, 10)
        # learning curves
        plt.subplot(grid[0, :6])
        plt.plot(np.arange(1, len(model.train_loss_history_) + 1), model.train_loss_history_)
        if algo not in ['ridge', 'lasso', 'elasticNet'] and early_stopping:
            plt.plot(np.arange(1, len(model.val_loss_history_) + 1), model.val_loss_history_)
        plt.plot(model.best_iteration_, model.best_loss_, 'r*')
        plt.xlabel('number of iterations')
        plt.ylabel('loss')
        if algo != 'ridge' and early_stopping and not early_stopping == 'built_in':
            plt.legend(['train. error', 'val. error', 'best model'])
            plt.title('best loss={:.3g} after {:d} iter - {} - {}\n'
                      'train r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                      'val   r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                      'test  r={:.3g} - R2={:.3g} - MSE={:.3g} - {:.2f} sec elapsed'
                      .format(model.best_loss_, model.best_iteration_, algo, str_params,
                              metrics[0, 0], metrics[0, 1], metrics[0, 2],
                              metrics[1, 0], metrics[1, 1], metrics[1, 2],
                              metrics[2, 0], metrics[2, 1], metrics[2, 2], model.run_time_), loc='left')
        else:
            plt.legend(['train. error', 'best model'])
            plt.title('best loss={:.3g} after {:d} iter - {} - {}\n'
                      'train r={:.3g} - R2={:.3g} - MSE={:.3g}\n'
                      'test  r={:.3g} - R2={:.3g} - MSE={:.3g} - {:.2f} sec elapsed'
                      .format(model.best_loss_, model.best_iteration_, algo, str_params, metrics[0, 0], metrics[0, 1],
                              metrics[0, 2], metrics[2, 0], metrics[2, 1], metrics[2, 2], model.run_time_), loc='left')
        # STRF
        plt.subplot(grid[0, 7:10])
        margin = np.max(np.abs([np.min(coefs), np.max(coefs)]))
        plt.imshow(coefs, aspect='auto', origin='lower', cmap='jet', extent=extent_tuple, clim=(-margin, margin))
        plt.colorbar()
        plt.xlabel('time lag (ms)')
        plt.ylabel(coefs_y_label + ' index')
        if n_target > 1:
            clim_margin = 1
            clim1 = (np.min([np.percentile(x, clim_margin) for x in target_sets]),
                     np.max([np.percentile(x, 100 - clim_margin) for x in target_sets]))
            clim2 = (np.min([np.percentile(x, clim_margin) for x in predicted_sets]),
                     np.max([np.percentile(x, 100 - clim_margin) for x in predicted_sets]))
            ax1 = plt.subplot(grid[1, :6])
            plt.imshow(target_sets[0].T, aspect='auto', origin='lower', cmap='jet', clim=clim1)
            plt.gca().axes.xaxis.set_ticklabels([])
            ax2 = plt.subplot(grid[2, :6])
            plt.imshow(predicted_sets[0].T, aspect='auto', origin='lower', cmap='jet', clim=clim2)
            plt.xlabel('time (samples)')
            plt.ylabel('frequency')
            plt.title('training set')
            plt.subplot(grid[1, 6:8], sharey=ax1)
            plt.imshow(target_sets[1].T, aspect='auto', origin='lower', cmap='jet', clim=clim1)
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.subplot(grid[2, 6:8], sharey=ax2)
            plt.imshow(predicted_sets[1].T, aspect='auto', origin='lower', cmap='jet', clim=clim2)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.title('validation set')
            plt.subplot(grid[1, 8:], sharey=ax1)
            plt.imshow(target_sets[2].T, aspect='auto', origin='lower', cmap='jet', clim=clim1)
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.subplot(grid[2, 8:], sharey=ax2)
            plt.imshow(predicted_sets[2].T, aspect='auto', origin='lower', cmap='jet', clim=clim2)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.title('test set')
        else:
            if model_type in ['decoding', 'recon'] and flag_log == 1:
                target_sets_plot = [np.exp(x) for x in target_sets]
                predicted_sets_plot = [np.exp(x) for x in predicted_sets]
            else:
                target_sets_plot = target_sets.copy()
                predicted_sets_plot = predicted_sets.copy()
            if early_stopping and not early_stopping == 'built_in':
                ax1 = plt.subplot(grid[1, :6])
                plt.plot(target_sets_plot[0])
                plt.plot(predicted_sets_plot[0])
                plt.xlim((0, len(predicted_sets_plot[0])))
                plt.legend(['actual', 'predicted'])
                plt.xlabel('time (samples)')
                plt.ylabel('amplitude (AU)')
                plt.title('training set')
                plt.subplot(grid[1, 6:8], sharey=ax1)
                plt.plot(target_sets_plot[1])
                plt.plot(predicted_sets_plot[1])
                plt.gca().axes.yaxis.set_ticklabels([])
                plt.title('validation set')
                plt.xlim((0, len(predicted_sets_plot[1])))
            else:
                ax1 = plt.subplot(grid[1, :8])
                plt.plot(target_sets_plot[0])
                plt.plot(predicted_sets_plot[0])
                plt.legend(['actual', 'predicted'])
                plt.xlabel('time (samples)')
                plt.ylabel('amplitude (AU)')
                plt.title('training set')
                plt.xlim((0, len(predicted_sets_plot[0])))
            plt.subplot(grid[1, 8:], sharey=ax1)
            plt.plot(target_sets_plot[-1])
            plt.plot(predicted_sets_plot[-1])
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.title('test set')
            plt.xlim((0, len(predicted_sets_plot[-1])))
            plt.show()
    return metrics, coefs, predicted_sets[-1], model.best_iteration_, model.run_time_, model.best_loss_


def get_activation_patterns(feature_sets, coefs, scaler):
    coefs_inv_scaled = scaler.inverse_transform(coefs.T)
    pattern = np.cov(feature_sets[0].T).dot(coefs_inv_scaled[0, :].dot(1.)).T
    return pattern


def check_r_sig(x, ci_thresh=0.9):
    p = 1 - ci_thresh
    n = len(x)
    z = np.arctanh(x)
    sem = np.std(z, ddof=1)
    ts = stats.t.ppf((p / 2, 1 - p / 2), n - 1)
    ci_z = z.mean() + ts * sem
    ci = np.tanh(ci_z)
    is_sig = ci[0] > 0
    return is_sig


########################################################################################################################
#  Model selection
########################################################################################################################
def grid_search(x, y, g, c, params, hyperparam_grid=None):
    n_cv = params['n_cv_tuning']
    hyperparam_list = ParameterGrid(hyperparam_grid)
    n_par = len(hyperparam_list)
    n_bon = params['n_best_of_n']
    param_verbose = params['verbose']
    param_fig = params['flag_fig']
    mdl_list = []
    cv_results_all = []
    if params['model_type'] == 'recon':
        idx_fixed_test = np.multiply(params['fixed_test'], params['fs']).astype('int16')
        idx_fixed_test[idx_fixed_test > len(g)] = len(g)
        if params['algo'] == 'MLP' and n_bon > 1:
            cv_results_all = np.zeros((n_cv, n_bon, n_par, 8))
    cv_results = np.zeros((n_cv, n_par, 8))
    print('Fitting {} folds for each of {} candidates, totalling {} fits'.format(n_cv, n_par, n_cv * n_par))
    idx = 0
    while idx < n_cv:
        print('[CV {}/{}]'.format(idx + 1, n_cv))
        split_indices = stratified_group_shuffle_split(g, c, **params)
        [feature_sets, target_sets] = perform_split(x, y, split_indices, **params)
        [feature_sets, scaler] = scale(feature_sets, **params)
        for idxPar in range(n_par):
            print('    [param {}/{}] {}'.format(idxPar + 1, n_par, hyperparam_list[idxPar]))
            params.update(hyperparam_list[idxPar])
            if params['algo'] == 'rMLRwES':
                mdl = RobustMultipleLinearRegressionEarlyStopping(**params)
                if params['early_stopping']:
                    mdl.fit(feature_sets[0], target_sets[0], feature_sets[1], target_sets[1])
                else:
                    mdl.fit(feature_sets[0], target_sets[0])
            elif params['algo'] in ['ridge', 'lasso', 'elasticNet']:
                mdl = RegularizedRegressionCustomEstimator(**params)
                mdl.fit(feature_sets[0], target_sets[0])
            elif params['algo'] == 'MLP':
                mdl = MLPRegressorCustomEarlyStopping(**params)
                mdl, mdl_list = best_of_n(mdl, feature_sets, target_sets, scaler, params)
            else:
                mdl = []
            [metrics, _, _, n_iter, runtime, _] = get_model_output(mdl, feature_sets, target_sets, scaler, **params)
            cv_results[idx, idxPar, :] = np.append(metrics[:2, :].ravel(), [n_iter, runtime])
            if params['model_type'] == 'recon':
                if params['algo'] == 'MLP' and n_bon > 1:
                    for idxBon in range(n_bon):
                        params['verbose'] = 0
                        params['flag_fig'] = 0
                        [metrics, _, _, n_iter, runtime, _] = get_model_output(mdl_list[idxBon], feature_sets,
                                                                               target_sets, scaler, **params)
                        cv_results_all[idx, idxBon, idxPar, :] = np.append(metrics[:2, :].ravel(), [n_iter, runtime])
                params['verbose'] = param_verbose
                params['flag_fig'] = param_fig
        if not np.all(np.isnan(cv_results[idx, [0, -1], 0])):
            idx += 1
    cv_results[np.isnan(cv_results[:, :, 3]), :] = np.nan
    idx_best_hyperparam = int(np.nanargmin(cv_results[:, :, 5], 1).mean().round())
    best_hyperparam = hyperparam_list[idx_best_hyperparam]
    if params['model_type'] == 'recon' and params['algo'] == 'MLP' and n_bon > 1:
        cv_results = cv_results_all
    return best_hyperparam, cv_results


def best_of_n(mdl, x, y, s, params):
    criterion = params['criterion_best_of_n'].split('_')
    criterion_idx_1 = 1
    criterion_idx_2 = 2
    if criterion[1] == 'train':
        criterion_idx_1 = 0
    elif criterion[1] == 'test':
        criterion_idx_1 = -1
    if criterion[0] == 'r':
        criterion_idx_2 = 0
        criterion_dir = 1
    elif criterion[0] == 'r2':
        criterion_idx_2 = 1
        criterion_dir = 1
    else:
        criterion_dir = -1
    best_metric = 0
    best_mdl = []
    mdl_list = []
    for idx_rep in range(params['n_best_of_n']):
        print('[best of {:d} - attempt #{:d}]'.format(params['n_best_of_n'], idx_rep + 1))
        mdl_tmp = deepcopy(mdl)
        if params['early_stopping']:
            mdl_tmp.fit(x[0], y[0], x[1], y[1])
        else:
            if criterion[1] == 'test':
                mdl_tmp.fit(np.concatenate(x[:2]), np.concatenate(y[:2]))
            else:
                mdl_tmp.fit(x[0], y[0])
        if params['n_best_of_n'] > 1:
            [metrics, _, _, _, _, _] = get_model_output(mdl_tmp, x, y, s, **params)
            metric_tmp = metrics[criterion_idx_1, criterion_idx_2]
            if idx_rep == 0 or criterion_dir * (best_metric - metric_tmp) < 0:
                best_metric = metric_tmp
                best_mdl = deepcopy(mdl_tmp)
        mdl_list.append(mdl_tmp)
    return best_mdl, mdl_list


########################################################################################################################
#  Visualizations
########################################################################################################################
def plot_prepared_data(x, y, c, g, model_type, n_feat_y, fs, flag_log, **kwargs):
    grid = plt.GridSpec(8, 1)
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(grid[:4])
    if model_type == 'encoding':
        plt.imshow(x[:, -n_feat_y:].T, aspect='auto', origin='lower',
                   clim=(np.percentile(x[:, -n_feat_y:], 5), np.percentile(x[:, -n_feat_y:], 95)))
    else:
        plt.imshow(x[:, :n_feat_y].T, aspect='auto', origin='lower',
                   clim=(np.percentile(x[:, :n_feat_y], 5), np.percentile(x[:, :n_feat_y], 95)))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.ylabel('features')
    ax2 = plt.subplot(grid[4:6], sharex=ax1)
    if flag_log == 1:
        y = np.exp(y)
    if y.shape[1] <= 1:
        plt.plot(y)
    else:
        plt.imshow(y.T, aspect='auto', origin='lower', clim=(np.percentile(y, 5), np.percentile(y, 95)))
    plt.xlim((0, x.shape[0]))
    plt.ylabel('target')
    ax3 = plt.subplot(grid[6:], sharex=ax1)
    plt.scatter(range(len(c)), [3.5] * len(c), c=c, cmap=plt.cm.Paired, lw=10, marker='_')
    c_map = ListedColormap(np.tile(plt.cm.Paired(range(12)), (int(np.ceil(g.max() / 12) * 12), 1))[:g.max() + 1, :])
    plt.scatter(range(len(g)), [.5] * len(g), c=g, cmap=c_map, lw=10, marker='_')
    ax3.set(ylim=[-1, 5], yticks=[.5, 3.5], yticklabels=['groups', 'classes'], xlim=[0, x.shape[0]],
            xticks=plt.gca().get_xticks()[1:-1], xticklabels=plt.gca().get_xticks()[1:-1] / fs, xlabel='time (s)')
    [plt.setp(x.get_xticklabels(), visible=False) for x in [ax1, ax2]]


def plot_splits(groups, split_indices, fs, n_splits=5, **kwargs):
    n_sets = len(split_indices[0])
    n_groups = len(groups)
    splits = np.ones((n_splits, n_groups))
    for idx in range(n_splits):
        for idxSet in range(1, n_sets):
            splits[idx, split_indices[idx][idxSet]] += idxSet
    plt.figure(figsize=(10, 10))
    plt.imshow(splits, aspect='auto', cmap=ListedColormap([u'#1f77b4', u'#ff7f0e', [.9, .8, 0]]))
    plt.gca().set(xticks=plt.gca().get_xticks()[1:-1], xticklabels=plt.gca().get_xticks()[1:-1] / fs,
                  xlabel='time (s)', ylabel='split (index)')


def plot_strf(coefs, r, r2, model_type, lags, offset, n_feat_y, **kwargs):
    strf_mean = coefs.mean(axis=0)
    strf = coefs.mean(axis=0) / coefs.std(axis=0, ddof=1)
    if model_type == 'encoding':
        extent_tuple = (round((offset - lags) * 1000), round(offset * 1000), 1, n_feat_y + 1)
    else:
        extent_tuple = (-round(offset * 1000), round((lags - offset) * 1000), 1, n_feat_y + 1)
    margin_mean = np.max(np.abs(strf_mean))
    margin = 5
    if check_r_sig(r):
        title_color = 'r'
    else:
        title_color = 'k'
    plt.figure(figsize=(16, 4))
    plt.subplot(131)
    plt.imshow(strf_mean, aspect='auto', origin='lower', cmap='jet', extent=extent_tuple,
               clim=(-margin_mean, margin_mean))
    plt.colorbar()
    plt.title('STRF (mean)')
    plt.subplot(132)
    plt.imshow(strf, aspect='auto', origin='lower', cmap='jet', extent=extent_tuple, clim=(-margin, margin))
    plt.colorbar()
    plt.title('STRF (Z)')
    plt.subplot(133)
    plt.hlines(0, 0, 3, color=(.7, .7, .7))
    plt.boxplot(np.vstack((r, r2)).T, notch=True, labels=('r', 'R2'))
    plt.title('r={:.3g} - R2={:.3g} - (N={})'.format(np.nanmean(r), r2.mean(), r.shape[0]), color=title_color)
    plt.ylim((-.2, 1))


def plot_y_pred(y_pred, y, flag_log, **kwargs):
    if flag_log == 1:
        y = np.exp(y)
        y_pred = np.exp(y_pred)
    mu = np.mean(y_pred, 0)
    sd = np.std(y_pred, 0)
    if len(mu.shape) > 1:
        mu = mu[:, 0]
        sd = sd[:, 0]
    plt.figure(figsize=(16, 4))
    plt.plot(y)
    plt.plot(mu, color='tab:orange')
    plt.fill_between(range(len(mu)), mu - sd, mu + sd, alpha=.3, color='tab:orange')
    plt.title('mu +/- sd ({} resamples)'.format(len(y_pred)))
