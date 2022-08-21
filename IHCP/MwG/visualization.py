import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as st
matplotlib.rcParams.update({'font.size': 14})

def CI_draw_all():
    def CI_draw(x_var, x_mean, label):
        Sigma = np.sqrt(x_var).flatten()

        # predicted expect and calculate confidence interval
        predicted_expect = x_mean.flatten()  # 预测值
        low_CI_bound, high_CI_bound = st.t.interval(0.95, data_points - 1,
                                                    loc=predicted_expect,
                                                    scale=Sigma)

        # plot confidence interval
        x = np.linspace(0, data_points - 1, num=data_points)

        plt.plot(predicted_expect, linewidth=3., label=label)
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2)

    x_true = np.array(pd.read_csv('data/qt_true.csv',header = None))
    for sigma_noise in [0.01,0.005,0.001]:#0.001,0.005,0.01
        x_mean_1 = np.load(f'data821/mwg_xmean_{sigma_noise}_N10000_M1.npy') # mh within gibbs
        x_var_1 =np.load(f'data821/mwg_xvar_{sigma_noise}_N10000_M1.npy')
        x_mean_2 = np.load(f'data821/tradition_xmean_{sigma_noise}.npy') # traditional
        x_var_2 =np.load(f'data821/tradition_xvar_{sigma_noise}.npy')
        x_mean_3 = np.load(f'data821/CwG_xmean_{sigma_noise}.npy')  # cvae with gibbs
        x_var_3 =np.load(f'data821/Cwg_xvar_{sigma_noise}.npy')


        # generate dataset
        data_points = 26
        sample_points = 50000
        Mu = x_true.flatten() # 真实值
        plt.plot(Mu, color='b', label='Ground truth')
        CI_draw(x_var_1, x_mean_1,'MwG')
        CI_draw(x_var_2, x_mean_2,'Baseline')
        CI_draw(x_var_3, x_mean_3,'CVAEwG')

        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$q(t)$')
        plt.ylim(-0.2, 1.2)
        plt.savefig(f'figs/IHCP_CI_{sigma_noise}.pdf')
        plt.show()



# 绘制theta的波动情况
def hyperparameter_estimate():
    for sigma_noise in [0.001,0.005,0.01]:
        theta1 = np.load(f'data821/mwg_theta_{sigma_noise}_N10000_M1.npy')
        theta2 = np.load(f'data821/tradition_theta_{sigma_noise}.npy')
        theta3 = np.load(f'data821/CwG_theta_{sigma_noise}.npy')
        plt.plot(theta1,label='MwG',alpha = 0.5)
        plt.plot(theta2,label='Baseline',alpha = 0.5)
        plt.plot(theta3,label='CVAEwG',alpha = 0.5)
        plt.xlabel('Iteration')
        plt.ylabel('$\hat{\sigma}_{obs}$')
        plt.axvline(5000,linestyle="--")
        plt.axhline(sigma_noise,linestyle="--")
        plt.legend(loc=1)
        plt.savefig(f'figs/IHCP_hyper_{sigma_noise}.pdf',bbox_inches = 'tight')
        plt.show()


def ACF():
    from statsmodels.tsa.stattools import acf
    import statsmodels.api as sm

    for sigma_noise in [0.001,0.005,0.01]:
        slice = 10
        X1 = np.load(f'data821/mwg_x_{sigma_noise}_N10000_M1.npy')[:,slice,:].flatten()
        X2 = np.load(f'data821/tradition_x_{sigma_noise}.npy')[:,slice]
        X3 = np.load(f'data821/CwG_x_{sigma_noise}.npy')[:,slice,:].flatten()

        fig = plt.figure(figsize=(12,10))
        ax1 = fig.add_subplot(311)
        sm.graphics.tsa.plot_acf(X1, lags=20,ax=ax1,title = 'Autocorrelation of MwG')
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

        ax2 = fig.add_subplot(312)
        sm.graphics.tsa.plot_acf(X2, lags=20,ax=ax2,title='Autocorrelation of baseline')
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

        ax3 = fig.add_subplot(313)
        sm.graphics.tsa.plot_acf(X3, lags=20,ax=ax3,title='Autocorrelation of CVAEwG')
        ax3.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

        plt.savefig(f'figs/IHCP_acfx_{sigma_noise}.pdf',bbox_inches = 'tight')
        plt.show()

    for sigma_noise in [0.001,0.005,0.01]:
        theta1 = np.load(f'data821/mwg_theta_{sigma_noise}_N10000_M1.npy')
        theta2 = np.load(f'data821/tradition_theta_{sigma_noise}.npy')
        theta3 = np.load(f'data821/CwG_theta_{sigma_noise}.npy')

        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(311)
        sm.graphics.tsa.plot_acf(theta1, lags=20, ax=ax1, title='Autocorrelation of MwG')
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

        ax2 = fig.add_subplot(312)
        sm.graphics.tsa.plot_acf(theta2, lags=20, ax=ax2, title='Autocorrelation of baseline')
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()

        ax3 = fig.add_subplot(313)
        sm.graphics.tsa.plot_acf(theta3, lags=20, ax=ax3, title='Autocorrelation of CVAEwG')
        ax3.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.savefig(f'figs/IHCP_acfTheta_{sigma_noise}.pdf', bbox_inches='tight')
        plt.show()