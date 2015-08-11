function [MI, R, LLRs, quantizer_boundaries, iter] = design_LLR_quant_samples(p0, L, num_iter, eps_stop, num_levels, symmetrize, boundaries_init)
%design_LLR_quant_samples
%   Finds optimal LLR quantizer that maximizes mutual information given LLR samples.
%
%   [MI, R, LLRs, quantizer_boundaries, iter] = ...
%       design_LLR_quant_samples(p0, L, num_iter, eps_stop, ...
%                                     num_levels, symmetrize, ...
%                                     boundaries_init)
%
%   Let X - L - Z be a Markov chain, where X is in {-1, +1}, L is the
%   posterior log-likelihood ratio (LLR) for X, and Z = Q(L) is a quantized
%   version of L. This MATLAB function finds a scalar quantizer Q that
%   maximizes the mutual information I(X; Z) given LLR samples L and the
%   prior probability P{X = +1}.
%   
%   Input arguments:
%     p0: prior probability of the binary data, p0 = P{X = +1} = 1 - P{X = -1}
%     L: vector of LLR samples, i.e., samples from the underlying unconditional LLR distribution
%     num_iter: maximum number of iterations to be performed
%     eps_stop: tolerance (stopping threshold), e.g., 1e-6
%     num_levels: number of quantization levels
%     symmetrize (optional argument): symmetrize = 1 forces the quantizer boundaries to be symmetric around zero (default: symmetrize = 0) 
%     boundaries_init (optional argument): initialization for the quantizer boundaries
%
%   Return values:
%     MI: estimate of the mutual information I(X; Z)
%     R: estimate of the entropy of the quantizer output, i.e., H(Z)
%     LLRs: the LLRs for X corresponding to the num_levels quantizer outputs
%     quantizer_boundaries: the optimized quantizer boundaries
%     iter: the number of iterations that have been performed
%
%   Example:
%     Consider equally likely X and conditionally Gaussian LLRs with
%     conditional mean mu equal to, say, 3. In this setting, an optimal
%     quantizer with, say, 6 quantization levels can be found as follows:
%
%     p0 = 0.5; mu = 3; num_iter = 50; eps_stop = 1e-6; num_levels = 6; n = 1e3;
%     X = 1 - 2 * randi([0, 1], 1, n);
%     L = X * mu + randn(1, n) * sqrt(2*mu);
%     [MI, R, LLRs, quantizer_boundaries, iter] = design_LLR_quant_samples(p0, L, num_iter, eps_stop, num_levels);
%
%     Note that the quantizer design does not need to know the data X.
%     Given the vector quantizer_boundaries, the quantizer can be
%     implemented using the function quantiz as follows:
%
%     [Z, LLRs_quant] = quantiz(L, quantizer_boundaries, LLRs);
%
%     Here, Z is the quantizer output in {0, 1, ..., num_levels - 1} and
%     L_quant are the corresponding quantized posterior LLRs for X.
%
%   See design_LLR_quant_distribution for a function that uses the
%   unconditional LLR distribution instead of LLR samples.
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
%   way use this code, please cite our original article as listed above.
%
%   Author: Andreas Winkelbauer <aw@andreaswinkelbauer.at>
%   Version: 1.0 (latest version: https://github.com/andreaswinkelbauer/spawc2015)
%   License: GPLv2 (https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt)

% ------------------------------------------------------------------------

    % ensure that we have the minimum number of arguments
    if (nargin < 5)
        error('ERROR: wrong number of input arguments');
    end
    
    % check the prior probability p0
    if (~isscalar(p0) || p0 <= 0 || p0 >= 1)
        error('ERROR: p0 must be a scalar in the interval (0, 1)');
    end
    
    % check LLR samples
    if (~isvector(L) || ~isreal(L) || any(~isfinite(L)))
        error('ERROR: L must be a real-valued vector');
    end
    num_samples = length(L);
    
    % ensure that we do at least one iteration
    if (~isscalar(num_iter) || num_iter < 1 || round(num_iter) ~= num_iter)
        error('ERROR: num_iter must be a positive integer');
    end
    
    % ensure that the stopping threshold is positive
    if (~isscalar(eps_stop) || eps_stop <= 0)
        error('ERROR: eps_stop must be a positive scalar');
    end
    
    % ensure that the number of quantization levels is at least 2
    if (~isscalar(num_levels) || num_levels < 2)
        error('ERROR: num_levels must be an integer >= 2');
    end
    
    % ensure that symmetrize is properly initialized
    if (nargin < 6 || isempty(symmetrize))
        symmetrize = 0;
    elseif (~isscalar(symmetrize) || (symmetrize ~= 0 && symmetrize ~= 1))
        error('ERROR: symmetrize must be either 0 (false) or 1 (true)');
    end
    
    % ensure that boundaries_init is properly initialized
    boundaries_init_computed = false;
    if (nargin < 7 || isempty(boundaries_init))
        % initialize the quantizer boundaries such that
        % the quantizer outputs are equally likely (maximum output entropy)
        boundaries_init_computed = true;
        boundaries_init = zeros(1, num_levels - 1);
        num_samples_per_bin = floor(num_samples / num_levels);
        L_samples_sorted = sort(L);

        for i = 1:(num_levels - 1)
            boundaries_init(i) = mean([L_samples_sorted(i*num_samples_per_bin) L_samples_sorted(i*num_samples_per_bin+1)]);
        end
    end

    % ensure that the initial quantizer boundaries are valid
    if (~isvector(boundaries_init) || ~isreal(boundaries_init) || numel(boundaries_init) ~= (num_levels - 1) || any(sort(boundaries_init) ~= boundaries_init) || any(~isfinite(boundaries_init)))
        if (boundaries_init_computed)
            error('ERROR: could not find initial boundaries (you may need to supply boundaries_init).');
        else
            error('ERROR: the supplied boundaries_init vector is invalid');
        end
    end
    quantizer_boundaries = boundaries_init(:).';

    % symmetrize quantizer boundaries if required
    if (symmetrize)
        quantizer_boundaries = symmetrize_boundaries(quantizer_boundaries);
    end

    % compute mutual information, entropy, and LLRs
    [MI, R, LLRs] = update_LLRs(L, quantizer_boundaries);

    % symmetrize quantized LLRs if required
    if (symmetrize)
        LLRs = symmetrize_LLRs(LLRs);
    end

    % initialize the iteration counter
    m = 1;
    
    % initialize eta which is used in the stopping criterion
    eta = inf;

    % alternatingly update quantizer boundaries and LLRs
    % until num_iter iterations are performed or 
    % the relative increase in mutual information is less than eps_stop
    while (m <= num_iter && eta >= eps_stop)
        % save of value of the mutual information
        MI_old = MI;
        
        % update the quantizer boundaries
        quantizer_boundaries = log(log((1 + exp(LLRs(2:end))) ./ (1 + exp(LLRs(1:(end-1))))) ./ log((1 + exp(-LLRs(1:(end-1)))) ./ (1 + exp(-LLRs(2:end)))));

        % symmetrize quantizer boundaries if required
        if (symmetrize)
            quantizer_boundaries = symmetrize_boundaries(quantizer_boundaries);
        end
        
        % update mutual information, entropy, and LLRs
        [MI, R, LLRs] = update_LLRs(L, quantizer_boundaries);
        
        % symmetrize quantized LLRs if required
        if (symmetrize)
            LLRs = symmetrize_LLRs(LLRs);
        end
    
        % compute the relative increase of the mutual information
        eta = (MI - MI_old) / MI;
        
        % increase the iteration counter
        m = m + 1;
    end
    
    % iter iterations have been performed
    iter = m-1;
end

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

% compute estimates of mutual information, entropy, and LLRs
function [MI, R, LLRs] = update_LLRs(L_samples, quantizer_boundaries)
    num_levels = length(quantizer_boundaries) + 1;
    num_samples = length(L_samples);
    
    LLRs = nan(1, num_levels);
    L_index = quantiz(L_samples, quantizer_boundaries);
    for i = 1:num_levels
        L = L_samples(L_index == i-1);
        LLRs(i) = log(sum(1 ./ (1 + exp(-L))) / sum(1 ./ (1 + exp(L))));
    end

    if (any(isnan(LLRs)))
        LLRs(isnan(LLRs)) = 0;
        LLRs = sort(LLRs);
    end

    if (any(LLRs ~= sort(LLRs)))
        error('ERROR: LLRs are not sorted. Too few samples?')
    end

    MI_tmp = zeros(2, num_levels);
    p_z = zeros(1, num_levels);
    for i = 1:num_levels
        MI_tmp(1, i) = log2((1 + exp(-LLRs(i))) / 2) * sum(1 ./ (1 + exp(-L_samples(L_index == i-1)))) / num_samples;
        MI_tmp(2, i) = log2((1 + exp(LLRs(i))) / 2) * sum(1 ./ (1 + exp(L_samples(L_index == i-1)))) / num_samples;
        p_z(i) = sum(L_index == i-1) / num_samples;
    end
    MI_tmp(isnan(MI_tmp)) = 0;
    MI = -sum(sum(MI_tmp));
    
    R = -sum(p_z(p_z>0) .* log2(p_z(p_z>0)));
end

% ------------------------------------------------------------------------

% return symmetric quantizer boundaries
function quantizer_boundaries_sym = symmetrize_boundaries(quantizer_boundaries)
    num_boundaries = length(quantizer_boundaries);
    quantizer_boundaries_sym = zeros(1, num_boundaries);

    lower_idx = 1:floor(num_boundaries / 2);
    upper_idx = num_boundaries:-1:(ceil(num_boundaries / 2) + 1);
    
    quantizer_boundaries_sym(upper_idx) = (quantizer_boundaries(upper_idx) - quantizer_boundaries(lower_idx)) / 2;
    quantizer_boundaries_sym(lower_idx) = -quantizer_boundaries_sym(upper_idx);
end

% ------------------------------------------------------------------------

% return a symmetric LLR vector
function LLRs_sym = symmetrize_LLRs(LLRs)
    num_levels = length(LLRs);
    LLRs_sym = zeros(1, num_levels);

    lower_idx = 1:floor(num_levels / 2);
    upper_idx = num_levels:-1:(ceil(num_levels / 2) + 1);
    
    LLRs_sym(upper_idx) = (LLRs(upper_idx) - LLRs(lower_idx)) / 2;
    LLRs_sym(lower_idx) = -LLRs_sym(upper_idx);
end
