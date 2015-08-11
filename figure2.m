function [num_samples, err_avg, err_95, err_99] = figure2()
%figure2
%   Reproduces the data which is required for Figure 2.
%
%   [num_samples, err_avg, err_95, err_99] = figure2()
%
%   Input arguments: none
%
%   Return values:
%     num_samples: number of LLR samples (x-axis in Figure 2)
%     err_avg: average relative error in percent
%     err_95: 95% relative error margin in percent
%     err_99: 99% relative error margin in percent
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
    mu = 5;
    fun_pL = @(L)((exp(-(L-mu).^2 / (4*mu)) + exp(-(L+mu).^2 / (4*mu))) / (4 * sqrt(pi * mu)));
    eps_stop = 1e-10;
    num_iter = 100;
    num_levels = 4;
    symmetrize = true;

    num_runs = 1e6;
    idx_95 = round(num_runs * 0.95);
    idx_99 = round(num_runs * 0.99);

    num_samples = ceil(logspace(1,3,21));

    MI1 = design_LLR_quant_distribution(p0, fun_pL, num_iter, eps_stop, num_levels, symmetrize);

    err_avg = zeros(length(num_samples), 1);
    err_95 = zeros(length(num_samples), 1);
    err_99 = zeros(length(num_samples), 1);

    for j = 1:length(num_samples)
        MI2 = zeros(num_runs, 1);
        
        for i = 1:num_runs
            bits_samples = 1 - 2 * randi([0, 1], num_samples(j), 1);
            L_samples = bits_samples * mu + randn(num_samples(j), 1) * sqrt(2*mu);
            [~, ~, ~, quantizer_boundaries] = design_LLR_quant_samples(p0, L_samples, num_iter, eps_stop, num_levels, symmetrize);
            MI2(i) = compute_mi(p0, fun_pL, quantizer_boundaries);
        end

        MI2_sort = sort(MI2, 'descend');

        err_avg(j) = 100 * (MI1 - mean(MI2)) / MI1;
        err_95(j) = 100 * (MI1 - MI2_sort(idx_95)) / MI1;
        err_99(j) = 100 * (MI1 - MI2_sort(idx_99)) / MI1;
    end
end

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

% compute mutual information
function MI = compute_mi(p0, p_L, quantizer_boundaries)
    num_levels = length(quantizer_boundaries) + 1;
    MI = 0;

    quantizer_boundaries = [-inf quantizer_boundaries inf];
    f0 = @(x)(p_L(x) ./ (1 + exp(-x)));
    f1 = @(x)(p_L(x) ./ (1 + exp(x)));

    for i = 1:num_levels
        v0 = integral(f0, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v1 = integral(f1, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v = v0 + v1;
        MI = MI + v0 * log2(v0 / (p0 * v)) + v1 * log2(v1 / ((1-p0) * v));
    end
end
