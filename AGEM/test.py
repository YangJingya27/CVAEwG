"""
Test script for reproducing Table 1, AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling, NeurIPS, 2019.

Copyright (C) 2019, Bichuan Guo <gbc16@mails.tsinghua.edu.cn>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from solver import solve_agem
import numpy as np

def run_test():
    x_dim = 50
    num = 100 #??

    sigma_prior = 0.01 # 对应新方法的 sigma_prior
    sigma_proposal = 0.01
    sigma_noise_hat_init = 0.01 # 初始项
    cov = np.identity(x_dim) * sigma_prior**2
    x_test = np.random.multivariate_normal([0]*x_dim, cov,num)
    x_true = x_test[:]

    noise_shape = x_true.shape[1:]
    n_dim = np.prod(noise_shape)


    for noise in [0.08]: #??   # 对应sigma_noise ，为真实的噪音取值！

        sigma_noise = [noise] * n_dim
        sigma_noise = np.array(sigma_noise[:n_dim]).reshape(noise_shape)

        rmse_mean, rmse_std, noise_mean, noise_std, variance_noise_hat_em = \
            solve_agem(x_true=x_true, sigma_noise=sigma_noise,
                       sigma_prior=sigma_prior, sigma_noise_hat_init=sigma_noise_hat_init,
                       sigma_proposal=sigma_proposal, type_proposal='mala',
                       candidate='mean', em_epochs=10, sample_epochs=1000)

        print('[AGEM] noise_gt: %.2f | rmse %.4f (%.4f), noise_est: %.4f (%.4f)' % (
            noise, rmse_mean, rmse_std, noise_mean, noise_std
        ))
        print(np.sqrt(variance_noise_hat_em))


if __name__ == '__main__':
    run_test()
