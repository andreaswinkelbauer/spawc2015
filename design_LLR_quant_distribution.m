function [MI, R, LLRs, quantizer_boundaries, iter] = design_LLR_quant_distribution(p0, fun_pL, num_iter, eps_stop, num_levels, symmetrize, boundaries_init)
%design_LLR_quant_distribution
%   Finds optimal LLR quantizer that maximizes mutual information given the LLR distribution.
%
%   [MI, R, LLRs, quantizer_boundaries, iter] = ...
%       design_LLR_quant_distribution(p0, fun_pL, num_iter, eps_stop, ...
%                                     num_levels, symmetrize, ...
%                                     boundaries_init)
%
%   Let X - L - Z be a Markov chain, where X is in {-1, +1}, L is the
%   posterior log-likelihood ratio (LLR) for X, and Z = Q(L) is a quantized
%   version of L. This MATLAB function finds a scalar quantizer Q that
%   maximizes the mutual information I(X; Z) given the LLR distribution
%   p(L) and the prior probability P{X = +1}.
%   
%   Input arguments:
%     p0: prior probability of the binary data, p0 = P{X = +1} = 1 - P{X = -1}
%     fun_pL: function handle to the unconditional LLR distribution (must accept vectors of arbitrary length as input)
%     num_iter: maximum number of iterations to be performed
%     eps_stop: tolerance (stopping threshold), e.g., 1e-6
%     num_levels: number of quantization levels
%     symmetrize (optional argument): symmetrize = 1 forces the quantizer boundaries to be symmetric around zero (default: symmetrize = 0) 
%     boundaries_init (optional argument): initialization for the quantizer boundaries
%
%   Return values:
%     MI: the mutual information I(X; Z)
%     R: the entropy of the quantizer output, i.e., H(Z)
%     LLRs: the LLRs for X corresponding to the num_levels quantizer outputs
%     quantizer_boundaries: the optimized quantizer boundaries
%     iter: the number of iterations that have been performed
%
%   Example:
%     Consider equally likely X and conditionally Gaussian LLRs with
%     conditional mean mu equal to, say, 3. In this setting, an optimal
%     quantizer with, say, 6 quantization levels can be found as follows:
%
%     p0 = 0.5; mu = 3; num_iter = 50; eps_stop = 1e-6; num_levels = 6;
%     fun_pL = @(L)((exp(-(L-mu).^2 / (4*mu)) + exp(-(L+mu).^2 / (4*mu))) / (4 * sqrt(pi * mu)));
%     [MI, R, LLRs, quantizer_boundaries, iter] = design_LLR_quant_distribution(p0, fun_pL, num_iter, eps_stop, num_levels);
%
%     Given the vector quantizer_boundaries, the quantizer can be
%     implemented using the function quantiz. For example:
%
%     n = 1e3;
%     X = 1 - 2 * randi([0, 1], 1, n);
%     L = X * mu + randn(1, n) * sqrt(2*mu);
%     [Z, L_quant] = quantiz(L, quantizer_boundaries, LLRs);
%
%     Here, Z is the quantizer output in {0, 1, ..., num_levels - 1} and
%     L_quant are the corresponding quantized posterior LLRs for X.
%
%   See design_LLR_quant_samples for a function that uses LLR samples
%   instead of the unconditional LLR distribution.
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

    % ensure that we have the minimum number of arguments
    if (nargin < 5)
        error('ERROR: wrong number of input arguments');
    end
    
    % check function handle
    p_L = fcnchk(fun_pL);

    % check the prior probability p0
    if (~isscalar(p0) || p0 <= 0 || p0 >= 1)
        error('ERROR: p0 must be a scalar in the interval (0, 1)');
    end
    
    % compute the normalization factor alpha
    % i.e., p_L / alpha is a probability density function
    alpha = integral(p_L, -inf, inf);
    
    % ensure that alpha is finite and positive
    % NOTE: we do not check if p_L is nonnegative everywhere
    if (~isfinite(alpha) || alpha <= 0)
        error('ERROR: fun_pL does not seem to be a probability density function');
    end

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
        for i = 1:(num_levels - 1)
            boundaries_init(i) = fzero(@(x)(integral(p_L, -inf, x) - i/num_levels), 0);
        end
    end

    % ensure that the initial quantizer boundaries are valid
    if (~isvector(boundaries_init) || numel(boundaries_init) ~= (num_levels - 1) || any(sort(boundaries_init) ~= boundaries_init) || any(~isfinite(boundaries_init)))
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
    [MI, R, LLRs] = update_LLRs(p0, p_L, alpha, quantizer_boundaries);

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
        [MI, R, LLRs] = update_LLRs(p0, p_L, alpha, quantizer_boundaries);
        
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

% compute mutual information, entropy, and LLRs
function [MI, R, LLRs] = update_LLRs(p0, p_L, alpha, quantizer_boundaries)
    num_levels = length(quantizer_boundaries) + 1;
    LLRs = zeros(1, num_levels);
    MI = 0;
    R = 0;

    quantizer_boundaries = [-inf quantizer_boundaries inf];
    f0 = @(x)(p_L(x) ./ (alpha * (1 + exp(-x))));
    f1 = @(x)(p_L(x) ./ (alpha * (1 + exp(x))));

    for i = 1:num_levels
        v0 = integral(f0, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v1 = integral(f1, quantizer_boundaries(i), quantizer_boundaries(i+1));
        v = v0 + v1;
        MI = MI + v0 * log2(v0 / (p0 * v)) + v1 * log2(v1 / ((1-p0) * v));
        R = R - v * log2(v);
        LLRs(i) = log(v0 / v1);
    end
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
