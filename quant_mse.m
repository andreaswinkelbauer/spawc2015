function [MI, R] = quant_mse(p0, fun_pL, num_iter, eps_stop, num_levels, symmetrize)
%quant_mse
%   Computes the rate and mutual information I(X; Z) of an MSE-optimal LLR quantizer.
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
    
    p_L = fcnchk(fun_pL);
    boundaries_init = zeros(1, num_levels - 1);
    for i = 1:(num_levels - 1)
        boundaries_init(i) = fzero(@(x)(integral(p_L, -inf, x) - i/num_levels), 0);
    end
    quantizer_boundaries = boundaries_init(:).';

    if (symmetrize)
        quantizer_boundaries = symmetrize_boundaries(quantizer_boundaries);
    end

    [MI, R] = compute_mi(p0, p_L, quantizer_boundaries);
    
    m = 1;
    eta = inf;

    while (m <= num_iter && eta >= eps_stop)
        quantizer_boundaries_old = quantizer_boundaries;
        quantizer_reproducers = compute_reproducers(p_L, quantizer_boundaries);
        quantizer_boundaries = mean([quantizer_reproducers(1:(end-1)); quantizer_reproducers(2:end)]);

        if (symmetrize)
            quantizer_boundaries = symmetrize_boundaries(quantizer_boundaries);
        end
        
        [MI, R] = compute_mi(p0, p_L, quantizer_boundaries);
        
        eta = max(abs(quantizer_boundaries_old - quantizer_boundaries)) / max(abs(quantizer_boundaries));
        m = m + 1;
    end
end

function quantizer_reproducers = compute_reproducers(p_L, quantizer_boundaries)
    num_levels = length(quantizer_boundaries) + 1;
    quantizer_reproducers = zeros(1, num_levels);
    quantizer_boundaries = [-inf quantizer_boundaries inf];
    f = @(x)(x .* p_L(x));
    
    for i = 1:num_levels
        quantizer_reproducers(i) = integral(f, quantizer_boundaries(i), quantizer_boundaries(i+1)) / integral(p_L, quantizer_boundaries(i), quantizer_boundaries(i+1));
    end
end

function [MI, R] = compute_mi(p0, p_L, quantizer_boundaries)
    num_levels = length(quantizer_boundaries) + 1;
    MI = 0;
    R = 0;

    quantizer_boundaries = [-inf quantizer_boundaries inf];
    f0 = @(x)(p_L(x) ./ (1 + exp(-x)));
    f1 = @(x)(p_L(x) ./ (1 + exp(x)));

    for i = 1:num_levels
        v0 = integral(f0, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v1 = integral(f1, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v = v0 + v1;
        MI = MI + v0 * log2(v0 / (p0 * v)) + v1 * log2(v1 / ((1-p0) * v));
        R = R - v * log2(v);
    end
end

function quantizer_boundaries_sym = symmetrize_boundaries(quantizer_boundaries)
    num_boundaries = length(quantizer_boundaries);
    quantizer_boundaries_sym = zeros(1, num_boundaries);

    lower_idx = 1:floor(num_boundaries / 2);
    upper_idx = num_boundaries:-1:(ceil(num_boundaries / 2) + 1);
    
    quantizer_boundaries_sym(upper_idx) = (quantizer_boundaries(upper_idx) - quantizer_boundaries(lower_idx)) / 2;
    quantizer_boundaries_sym(lower_idx) = -quantizer_boundaries_sym(upper_idx);
end
