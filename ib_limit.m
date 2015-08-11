function [MI, R] = ib_limit(Px, Py_x, beta, eps_stop, max_iter)
%ib_limit
%   Computes the information-rate function using the information bottleneck algorithm.
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

    J = size(Py_x, 1);
    M = size(Py_x, 2);
    K = M + 1;

    Pxy  = (Px * ones(1, M)) .* Py_x;
    Py   = sum(Pxy);
    idx = Py > 0;
    Py_x = Py_x(:, idx);

    % compute joint pdf Pxy
    Pxy  = (Px * ones(1, M)) .* Py_x;
    Py   = sum(Pxy);
    Px_y = Pxy ./ (ones(J, 1) * Py);
    
    % initialize p(z | y)
    Pz_y = rand(K, M);
    Pz_y = Pz_y ./ (ones(K, 1) * sum(Pz_y));

    % iteration counter
    k = 1;

    % initialize variables
    eta = inf;
    d_avg = inf;
    
    while ((eta >= eps_stop) && (k <= max_iter))
        % save the average distortion
        d_avg_old = d_avg;

        % calculate p(z)
        Pz = sum(Pz_y .* (ones(K,1) * Py), 2).';

        % calculate p(x | z)
        Px_z = Pxy * Pz_y.' ./ (ones(J, 1) * Pz);
        Px_z(~isfinite(Px_z)) = 1 / J;
        
        % calculate d(y, z)
        d1 = (ones(K, 1) * Px_y(1, :)) .* log2((ones(K, 1) * Px_y(1, :)) ./ (Px_z(1, :).' * ones(1, M)));
        d2 = (ones(K, 1) * Px_y(2, :)) .* log2((ones(K, 1) * Px_y(2, :)) ./ (Px_z(2, :).' * ones(1, M)));
        
        d1(isnan(d1)) = 0;
        d2(isnan(d2)) = 0;
        
        d = d1 + d2;
        
        % update Pz_y
        Pz_y = (Pz.' * ones(1, M)) .* exp(-beta * d);
        Pz_y = Pz_y ./ (ones(K, 1) * sum(Pz_y));
        
        % compute average d
        d_avg = sum(sum(d .* Pz_y) .* Py);
        
        % compute eta
        eta = (d_avg_old - d_avg) / d_avg;

        % increase the iteration counter
        k = k + 1;
    end
    
    % calculate p(z)
    Pz = sum(Pz_y .* (ones(K,1) * Py), 2).';

    % calculate p(x | z)
    Px_z = Pxy * Pz_y.' ./ (ones(J, 1) * Pz);
    Px_z(~isfinite(Px_z)) = 1 / J;

    MI = sum(sum(Px_z .* (ones(J, 1) * Pz) .* log2(Px_z ./ (Px * ones(1, K)))));
    t = Pz_y .* log2(Pz_y);
    t(isnan(t)) = 0;
    R  = -sum(Pz .* log2(Pz)) + sum(sum(t) .* Py);
end
