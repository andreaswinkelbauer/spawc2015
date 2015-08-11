function [I, R] = figure3()
%figure3
%   Reproduces the data which is required for Figure 3.
%
%   [I, R] = figure3()
%
%   Input arguments: none
%
%   Return values:
%     I: cell array containing the y-axis values (mutual information)
%     R: cell array containing the x-axis values (entropy)
%
% ------------------------------------------------------------------------
%
%   Reference:
%   [1] A. Winkelbauer and G. Matz, On Quantization of Log-Likelihood Ratios
%       for Maximum Mutual Information, in Proc. 16th IEEE Int. Workshop on
%       Signal Processing Advances in Wireless Communications (SPAWC 2015),
%       June 2015, Stockholm (Sweden).
%
%   BibTeX:
%   @InProceedings{winkelbauer2015a,
%     Title = {On Quantization of Log-Likelihood Ratios for Maximum Mutual Information},
%     Author = {Winkelbauer, Andreas and Matz, Gerald},
%     Booktitle = {Proc. 16th IEEE Int. Workshop on Signal Processing Advances in Wireless Communications (SPAWC 2015)},
%     Year = {2015},
%     Month = jun
%   }
%
%   License: This code is licensed under the GPLv2 license. If you in any
%   way use this code for research that results in publications, please
%   cite our original article as indicated above.
%
%   Author: Andreas Winkelbauer <aw@andreaswinkelbauer.at>
%   Version: 1.0 (latest version: https://github.com/andreaswinkelbauer/spawc2015)
%   License: GPLv2 (https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt)

% ------------------------------------------------------------------------

    p0 = 0.5;
    mu = [1 5 10];
    eps_stop = 1e-10;
    num_iter = 100;
    num_levels = 2:8;
    symmetrize = true;
    N = 101;
    beta = logspace(-6, 2, 100);
    
    I_q_MI  = zeros(length(mu), length(num_levels));
    I_q_MSE = zeros(length(mu), length(num_levels));
    I_limit = zeros(length(mu), length(beta));
    R_q_MI  = zeros(length(mu), length(num_levels));
    R_q_MSE = zeros(length(mu), length(num_levels));
    R_limit = zeros(length(mu), length(beta));

    for i = 1:length(mu)
        fun_pL = @(L)((exp(-(L-mu(i)).^2 / (4*mu(i))) + exp(-(L+mu(i)).^2 / (4*mu(i)))) / (4 * sqrt(pi * mu(i))));
        for j = 1:length(num_levels)
            [I_q_MI(i, j), R_q_MI(i, j)] = design_LLR_quant_distribution(p0, fun_pL, num_iter, eps_stop, num_levels(j), symmetrize);
            [I_q_MSE(i, j), R_q_MSE(i, j)] = quant_mse(p0, fun_pL, num_iter, eps_stop, num_levels(j), symmetrize);
        end
        
        L_max = mu(i) + 5 * sqrt(2*mu(i));
        y = linspace(-L_max, L_max, N);

        Py_x = [qfunc((y(1:(end-1)) - mu(i))/sqrt(2*mu(i))) - qfunc((y(2:end) - mu(i))/sqrt(2*mu(i))); ...
                qfunc((y(1:(end-1)) + mu(i))/sqrt(2*mu(i))) - qfunc((y(2:end) + mu(i))/sqrt(2*mu(i)))];
            
        for j = 1:length(beta)
            [I_limit(i, j), R_limit(i, j)] = ib_limit([p0; 1-p0], Py_x, beta(j), eps_stop, num_iter);
        end
    end
    
    I = {I_limit I_q_MI I_q_MSE};
    R = {R_limit R_q_MI R_q_MSE};
end
