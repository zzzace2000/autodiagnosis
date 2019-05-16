import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('../')

from arch.MGP_RNN import MGP_RNN


class LMC_MGP_RNN(MGP_RNN):
    def __init__(self, init_lmc_lengths=None, **kwargs):
        if init_lmc_lengths is None:
            init_lmc_lengths = [0.5, 3, 12]

        num_lmc = len(init_lmc_lengths)
        self.num_lmc = num_lmc
        self.init_lmc_lengths = init_lmc_lengths
        super(LMC_MGP_RNN, self).__init__(**kwargs)

    def create_gp_params(self):
        if self.init_lmc_lengths is not None:
            initializer = np.log(self.init_lmc_lengths).astype(np.float32)
            log_lengths = tf.get_variable("GP-log-lengths", dtype=tf.float32, initializer=initializer)
        else:
            # in fully separable case all labs share same time-covariance. Start with 4
            log_lengths = tf.Variable(tf.random_normal([self.num_lmc], mean=1.35, stddev=0.1),
                                      name="GP-log-length")
        self.lengths = tf.exp(log_lengths)

        # init kernel weights
        # for i in range(self.num_lmc):
        #     weight = tf.Variable(tf.ones((self.num_features, 1)))
        #     self.kernel_weights.append(weight)

        # different noise level of each lab
        log_noises = tf.Variable(tf.random_normal([self.num_features], mean=-2, stddev=0.1),
                                 name="GP-log-noises")
        self.noises = tf.exp(log_noises)

        # init cov between labs
        self.Kfs = []
        # self.kernel_weights = []
        for i in range(self.num_lmc):
            L_f_init = tf.Variable(tf.eye(self.num_features), name="GP-Lf")
            # Lower triangular part of matrix
            Lf = tf.matrix_band_part(L_f_init, -1, 0)
            Kf = tf.matmul(Lf, tf.transpose(Lf))
            self.Kfs.append(Kf)

    def draw_GP(self, Yi, Ti, Xi, ind_kfi, ind_kti):
        """
        given GP hyperparams and data values at observation times, draw from
        conditional GP

        inputs:
            length, noises, Lf, Kf: GP params
            Yi: observation values
            Ti: observation times
            Xi: grid points (new times for rnn)
            ind_kfi, ind_kti: indices into Y

        returns:
            draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
        """
        # Cache some calculations
        grid_f = tf.meshgrid(ind_kfi, ind_kfi)  # same as np.meshgrid
        grid_t = tf.meshgrid(ind_kti, ind_kti)

        D = tf.diag(self.noises)
        DI_big = tf.gather_nd(D, tf.stack((grid_f[0], grid_f[1]), -1))
        DI = tf.diag(tf.diag_part(DI_big))  # D kron I

        # K_weights = [tf.matmul(kw, tf.transpose(kw)) for kw in self.kernel_weights]

        ky_tmp = []
        for lmc_idx in range(self.num_lmc):
            K_tt = self.OU_kernel(self.lengths[lmc_idx], Ti, Ti)

            Kf_big = tf.gather_nd(self.Kfs[lmc_idx], tf.stack((grid_f[0], grid_f[1]), -1))

            Kt_big = tf.gather_nd(K_tt, tf.stack((grid_t[0], grid_t[1]), -1))

            Kf_Ktt = tf.multiply(Kf_big, Kt_big)

            ky_tmp.append(Kf_Ktt)

        # data covariance.
        ny = tf.shape(Yi)[0]
        Ky = sum(ky_tmp) + DI + 1e-6 * tf.eye(ny)

        ### build out cross-covariances and covariance at grid
        nx = tf.shape(Xi)[0]

        # Cache calculations
        ind = tf.concat([tf.tile([i], [nx]) for i in range(self.num_features)], 0)
        grid = tf.meshgrid(ind, ind)
        ind2 = tf.tile(tf.range(nx), [self.num_features])
        grid2 = tf.meshgrid(ind2, ind2)
        full_f = tf.concat([tf.tile([i], [nx]) for i in range(self.num_features)], 0)
        grid_1 = tf.meshgrid(full_f, ind_kfi, indexing='ij')
        full_x = tf.tile(tf.range(nx), [self.num_features])
        grid_2 = tf.meshgrid(full_x, ind_kti, indexing='ij')

        K_ff_tmp = []
        K_fy_tmp = []
        for lmc_idx in range(self.num_lmc):
            K_xx = self.OU_kernel(self.lengths[lmc_idx], Xi, Xi)
            K_xt = self.OU_kernel(self.lengths[lmc_idx], Xi, Ti)

            Kf_big = tf.gather_nd(self.Kfs[lmc_idx], tf.stack((grid[0], grid[1]), -1))
            Kxx_big = tf.gather_nd(K_xx, tf.stack((grid2[0], grid2[1]), -1))

            K_ff = tf.multiply(Kf_big, Kxx_big)  # cov at grid points
            K_ff_tmp.append(K_ff)

            Kf_big = tf.gather_nd(self.Kfs[lmc_idx], tf.stack((grid_1[0], grid_1[1]), -1))

            Kxt_big = tf.gather_nd(K_xt, tf.stack((grid_2[0], grid_2[1]), -1))

            K_fy = tf.multiply(Kf_big, Kxt_big)
            K_fy_tmp.append(K_fy)

        K_ff = sum(K_ff_tmp)
        K_fy = sum(K_fy_tmp)

        ## Calculate the ultimate GP dist
        y_ = tf.reshape(Yi, [-1, 1])
        Ly = tf.cholesky(Ky)
        Mu = tf.matmul(K_fy, tf.cholesky_solve(Ly, y_))

        xi = tf.random_normal((nx * self.num_features, self.n_mc_smps))
        Sigma = K_ff - tf.matmul(K_fy, tf.cholesky_solve(Ly, tf.transpose(K_fy))) + \
                1e-6 * tf.eye(tf.shape(K_ff)[0])
        draw = Mu + tf.matmul(tf.cholesky(Sigma), xi)
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),
                                               [self.n_mc_smps, self.num_features, nx]),
                                    perm=[0, 2, 1])

        return draw_reshape

    def _monitor_params(self):
        params = {'gp_length': self.lengths}
        # for i in range(self.num_lmc):
        #     params['w_%d' % i] = self.kernel_weights[i]
        return params
