function K = covDrift(cov, hyp, x, z, i)

% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

% report number of parameters
if (nargin < 3)
  K = ['2+' feval(cov{:})];
  return;
end

% make sure z exists
if (nargin < 4)
  z = []; 
end

% determine mode
xeqz = (numel(z) == 0); 
diagonal = (strcmp(z, 'diag') && (numel(z) > 0));
derivatives = (nargin == 5);

begin_time = hyp(1);
end_time   = hyp(2);
others     = hyp(3:end);

% find points that are within the interval of interest
ind_x = (x(:, end) >= begin_time) & (x(:, end) <= end_time);

% find covariances between points and changepoint boundaries
   c = [begin_time; end_time];
K_cc = feval(cov{:}, others, c);
K_cc = improve_covariance_conditioning(K_cc);
K_xc = feval(cov{:}, others, x, c);

% vector kxx
if (diagonal)
  if (~derivatives)
    K = feval(cov{:}, others, x, 'diag');
  else
    K = feval(cov{:}, others, x, 'diag', i - 2);
  end

  % fix K to 0 outside [begin_time, end_time]
  K(~ind_x) = 0;
  return;
else
  % symmetric matrix Kxx
  if (xeqz)
    if (~derivatives)
      K = feval(cov{:}, others, x);
      K2 = (K_xc / K_cc) * K_xc';
      
      K = K - K2;
      % hack for numerical stuff
      K(diag_inds(K)) = max(diag(K), eps);
    else
      % derivatives wrt begin and end time
      if (i < 3)
        K = zeros(size(x, 1));
        return;
      end
      % derivatives with respect to underlying covariance

      % first part
      DK = feval(cov{:}, others, x, [], i - 2);

      % second part
      DK_xc = feval(cov{:}, others, x, c, i - 2);
      DK_cc = feval(cov{:}, others, c, [], i - 2);
        DK2 = (DK_xc / K_cc) *   K_xc' + ...
              ( K_xc / K_cc) *  DK_xc' - ...
              ( K_xc / K_cc) * (DK_cc / DK_cc) * K_xc';
      K = DK - DK2;
    end

    % fix K to 0 outside [begin_time, end_time]
    K(:, ~ind_x) = 0;
    K(~ind_x, :) = 0;
    return;

  % cross covariances Kxz
  else                                                   
    % find points that are within the interval of interest
    ind_z = (z >= begin_time) & (z <= end_time);

    if (~derivatives)
         K = feval(cov{:}, others, x, z);
      K_cz = feval(cov{:}, others, c, z);
      K2 = (K_xc / K_cc) * K_cz;
      K = K - K2;
    else
      if (i < 3)
        K = zeros(size(x, 1), size(z, 1));
        return;
      end

      % first part
      DK = feval(cov{:}, others, x, z, i - 2);

      % second part
      DK_xc = feval(cov{:}, others, x, c, i - 2);
      DK_cc = feval(cov{:}, others, c, [], i - 2);
       K_cz = feval(cov{:}, others, c, z);
      DK_cz = feval(cov{:}, others, c, z, i - 2);
        DK2 = (DK_xc / K_cc) *  K_cz + ...
              ( K_xc / K_cc) * DK_cz - ...
              ( K_xc / K_cc) * (DK_cc / DK_cc) * K_cz;
      K = DK - DK2;
    end

    % fix K to 0 outside [begin_time, end_time]
    K(~ind_x, :) = 0;
    K(:, ~ind_z) = 0;
  end
end