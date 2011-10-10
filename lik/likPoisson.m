function [varargout] = likPoisson(hyp, y, mu, s2, inf, i)
  
if (nargin < 2)
  % report number of hyperparameters
  varargout = { '1' }; 
  return; 
end

z = hyp;
if (numel(y) == 0)
  y = ones(size(mu));
end

if (nargin < 5)
  
  no_sigma = (numel(s2) == 0) || ((numel(s2) == 1) && (s2 == 0));
  
  if (no_sigma)
    lambda = z * exp(mu) .* y;
    lp = -lambda + log(lambda) .* y - gammaln(y + 1);
  else
    lp = zeros(size(y));
    for i = 1:length(mu)
      r = mu(i) + sqrt(s2(i)) * randn(10000, 1);
      lambdas = z * exp(r) .* y(i);
      lp(i) = mean(-lambdas + log(lambdas) * y(i) - gammaln(y(i) + 1));
    end
  end    
 
  if (nargout > 1)
    ymu = zeros(size(mu));
    for i = 1:length(ymu)
      min_range = mu(i) - 6 * sqrt(s2(i));
      max_range = mu(i) + 6 * sqrt(s2(i));
      
      integral = @(f) exp(log(z) + f + norm_lpdf(f, mu(i), sqrt(s2(i))));
      ymu(i) = quadgk(integral, min_range, max_range);
    end
    if (nargout > 2)
      ys2 = s2;
    end
  end

  varargout = {lp, ymu, ys2};
else                                                            % inference mode
  %switch inf 
    %case 'infLaplace'
      if (nargin == 5)
        f = mu;
        lambda = z * exp(f) .* ones(size(y));
        lp = -lambda + log(lambda) .* y - gammaln(y + 1);
        dlp = [];
        d2lp = [];
        d3lp = [];
        if (nargout > 1)
          dlp = y - z * exp(f);
          if (nargout > 2)
            d2lp = dlp - y;
            if (nargout > 3)
              d3lp = d2lp;
            end
          end
        end
        varargout = {sum(lp), dlp, d2lp, d3lp};
      else
        varargout = {zeros(size(y)), zeros(size(y))};
      end
  %end
end
