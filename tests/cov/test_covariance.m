function passed = test_covariance(covariance, D, num_train, num_test)
  
  passed = true;
  tolerance = 1e-12;
  name = functions(covariance);
  name = name.function;
  
  eval(['GPMLcovariance = @GPML' name ';']);
  
  disp(['testing interface (1), s = ' name]);
  s = feval(covariance);
  disp(['   s = ' name '; s = ' s]);
 
  num_hyp = eval(s);
  hyp = rand(num_hyp, 1);
  x = rand(num_train, D);
  
  disp(['testing interface (2), K = ' name '(hyp, x)']);
  K = covariance(hyp, x);
  disp(['   K = ' name '(hyp, x)']);
  disp('testing for consistency with GPML implementation');
  K_GPML = GPMLcovariance(hyp, x);
  disp(['   K_GPML = GPML' name '(hyp, x)']);
  error = norm(K - K_GPML);
  disp(['   norm(K - K_GPML) = ' num2str(error)]);
  passed = passfail(error < tolerance) && passed;

  disp(['testing interface (3), K2 = ' name '(hyp, x, []), should equal K']);
  K2 = covariance(hyp, x, []);
  disp(['   K2 = ' name '(hyp, x, [])']);
  error = max(abs(K2(:) - K(:)));
  disp(['   max(abs(K2(:) - K(:))) = ' num2str(error)]);
  passed = passfail(error == 0) && passed;

  xs = rand(num_test, D);

  disp(['testing interface (4), Ks = ' name '(hyp, x, xs)']);
  Ks = covariance(hyp, x, xs);
  disp(['   Ks = ' name '(hyp, x, xs)']);
  disp('testing for consistency with GPML implementation');
  Ks_GPML = GPMLcovariance(hyp, x);
  disp(['   Ks_GPML = GPML' name '(hyp, x, xs)']);
  error = norm(K - Ks_GPML);
  disp(['   norm(K - Ks_GPML) = ' num2str(error)]);
  passed = passfail(error < tolerance) && passed;

  disp(['testing interface (5), Kss = ' name '(hyp, xs, ''diag'')']);
  Kss = covariance(hyp, xs, 'diag');
  disp(['   Kss = ' name '(hyp, xs, ''diag'')']);
  disp('testing for consistency with GPML implementation');
  Kss_GPML = GPMLcovariance(hyp, xs, 'diag');
  disp(['   Kss_GPML = GPML' name '(hyp, xs, ''diag'')']);
  error = norm(Kss - Kss_GPML);
  disp(['   norm(Kss - Kss_GPML) = ' num2str(error)]);
  passed = passfail(error < tolerance) && passed;

  disp(['testing interface (6), dKi = ' name '(hyp, x, [], i)']);
  for i = 1:num_hyp
    disp(['   testing derivative #' num2str(i)]);
    dKi = covariance(hyp, x, [], i);
    disp(['      dKi = ' name '(hyp, x, [], ' num2str(i) ')']);
    disp('   testing for consistency with GPML implementation');
    dKi_GPML = GPMLcovariance(hyp, x, [], i);
    disp(['      dKi_GPML = GPML' name '(hyp, x, [], ' num2str(i) ')']);
    error = norm(dKi - dKi_GPML);
    disp(['      norm(dKi - dKi_GPML) = ' num2str(error)]);
    passed = passfail(error < tolerance) && passed;
  end
  
  disp(['testing interface (7), dKsi = ' name '(hyp, x, xs, i)']);
  for i = 1:num_hyp
    disp(['   testing derivative #' num2str(i)]);
    dKsi = covariance(hyp, x, xs, i);
    disp(['      dKsi = ' name '(hyp, x, xs, ' num2str(i) ')']);
    disp('   testing for consistency with GPML implementation');
    dKsi_GPML = GPMLcovariance(hyp, x, xs, i);
    disp(['      dKsi_GPML = GPML' name '(hyp, x, xs, ' num2str(i) ')']);
    error = norm(dKsi - dKsi_GPML);
    disp(['      norm(dKsi - dKsi_GPML) = ' num2str(error)]);
    passed = passfail(error < tolerance) && passed;
  end
  
  disp(['testing interface (8), dKssi = ' name '(hyp, x, ''diag'', i)']);
  for i = 1:num_hyp
    disp(['   testing derivative #' num2str(i)]);
    dKssi = covariance(hyp, x, 'diag', i);
    disp(['      dKssi = ' name '(hyp, x, ''diag'', ' num2str(i) ')']);
    disp('   testing for consistency with GPML implementation');
    dKssi_GPML = GPMLcovariance(hyp, x, 'diag', i);
    disp(['      dKssi_GPML = GPML' name '(hyp, x, ''diag'', ' num2str(i) ')']);
    error = norm(dKssi - dKssi_GPML);
    disp(['      norm(dKssi - dKssi_GPML) = ' num2str(error)]);
    passed = passfail(error < tolerance) && passed;
  end
  
  if (passed)
    disp('All tests passed.');
  else
    disp('Not all tests passed!');
    return;
  end
  
  disp('testing timing');
  tic;
  for i = 1:100
    K = covariance(hyp, x);
  end
  elapsed = toc;
  disp(['   average time to calculate compiled covariance (K = ' name '(hyp, x), 100 runs) = ' num2str(elapsed / 100) 's.']);

  for i = 1:100
    K = GPMLcovariance(hyp, x);
  end
  elapsed = toc;
  disp(['   average time to calculate GPML covariance (K = GPML' name '(hyp, x), 100 runs) = ' num2str(elapsed / 100) 's.']);
end

function passed = passfail(passed)
  if (passed)
    disp('PASS.');
  else
    disp('FAIL!');
  end
end