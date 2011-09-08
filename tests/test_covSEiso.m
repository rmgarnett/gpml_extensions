tolerance = 1e-12;

failed = false;

disp('testing interface (1), s = covNAME, covSEiso should return 2');
s = covSEiso;
disp(['   s = covSEiso; s = ' num2str(s)]);
if (s == '2')
  disp('PASS.');
else
  failed = true;
  disp('FAIL!');
end

hyp = [0; 0];
x = rand(100, 10);

disp('testing interface (2), K = covNAME(hyp, x)');
K = covSEiso(hyp, x);
disp('   K = covSEiso(hyp, x)');
disp('testing for consistency with GPML implementation');
K_GPML = GPMLcovSEiso(hyp, x);
disp('   K_GPML = GPMLcovSEiso(hyp, x)');
error = norm(K - K_GPML);
disp(['   norm(K - K_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('PASS.');
else
  failed = true;
  disp('FAIL!');
end

disp('testing interface (3), K2 = covNAME(hyp, x, []), should equal K');
K2 = covSEiso(hyp, x, []);
disp('   K2 = covSEiso(hyp, x, [])')
error = max(abs(K2(:) - K(:)));
disp(['   max(abs(K2(:) - K(:))) = ' num2str(error)]);
if (error == 0)
  disp('PASS.');
else
  failed = true;
  disp('FAIL!');
end

xs = rand(1000, 10);

disp('testing interface (4), Ks = covName(hyp, x, xs)');
Ks = covSEiso(hyp, x, xs);
disp('   Ks = covSEiso(hyp, x, xs)');
disp('testing for consistency with GPML implementation');
Ks_GPML = GPMLcovSEiso(hyp, x);
disp('   Ks_GPML = GPMLcovSEiso(hyp, x, xs)');
error = norm(K - Ks_GPML);
disp(['   norm(K - Ks_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('PASS.');
else
  failed = true;
  disp('FAIL!');
end

disp('testing interface (5), Kss = covNAME(hyp, xs, ''diag'')');
Kss = covSEiso(hyp, xs, 'diag');
disp('   Kss = covSEiso(hyp, xs, ''diag'')');
disp('testing for consistency with GPML implementation');
Kss_GPML = GPMLcovSEiso(hyp, xs, 'diag');
disp('   Kss_GPML = GPMLcovSEiso(hyp, xs, ''diag'')');
error = norm(Kss - Kss_GPML);
disp(['   norm(Kss - Kss_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('PASS.');
else
  failed = true;
  disp('FAIL!');
end

disp('testing interface (6), dKi = covName(hyp, x, [], i)');
disp('   testing input scale derivative (i = 1)');
dKi = covSEiso(hyp, x, [], 1);
disp('      dKi = covSEiso(hyp, x, [], 1)');
disp('   testing for consistency with GPML implementation');
dKi_GPML = GPMLcovSEiso(hyp, x, [], 1);
disp('      dKi_GPML = GPMLcovSEiso(hyp, x, [], 1)');
error = norm(dKi - dKi_GPML);
disp(['      norm(dKi - dKi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end
disp('   testing output scale derivative (i = 2)');
dKi = covSEiso(hyp, x, [], 2);
disp('      dKi = covSEiso(hyp, x, [], 2)');
disp('   testing for consistency with GPML implementation');
dKi_GPML = GPMLcovSEiso(hyp, x, [], 2);
disp('      dKi_GPML = GPMLcovSEiso(hyp, x, [], 2)');
error = norm(dKi - dKi_GPML);
disp(['      norm(dKi - dKi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end

disp('testing interface (7), dKsi = covName(hyp, x, xs, i)');
disp('   testing input scale derivative (i = 1)');
dKsi = covSEiso(hyp, x, xs, 1);
disp('      dKsi = covSEiso(hyp, x, xs, 1)');
disp('   testing for consistency with GPML implementation');
dKsi_GPML = GPMLcovSEiso(hyp, x, xs, 1);
disp('      dKsi_GPML = GPMLcovSEiso(hyp, x, xs, 1)');
error = norm(dKsi - dKsi_GPML);
disp(['      norm(dKsi - dKsi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end
disp('   testing output scale derivative (i = 2)');
dKsi = covSEiso(hyp, x, xs, 2);
disp('      dKsi = covSEiso(hyp, x, xs, 2)');
disp('   testing for consistency with GPML implementation');
dKsi_GPML = GPMLcovSEiso(hyp, x, xs, 2);
disp('      dKsi_GPML = GPMLcovSEiso(hyp, x, xs, 2)');
error = norm(dKsi - dKsi_GPML);
disp(['      norm(dKsi - dKsi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end

disp('testing interface (8), dKsi = covName(hyp, x, ''diag'', i)');
disp('   testing input scale derivative (i = 1)');
dKssi = covSEiso(hyp, x, 'diag', 1);
disp('      dKssi = covSEiso(hyp, x, ''diag'', 1)');
disp('   testing for consistency with GPML implementation');
dKssi_GPML = GPMLcovSEiso(hyp, x, 'diag', 1);
disp('      dKssi_GPML = GPMLcovSEiso(hyp, x, ''diag'', 1)');
error = norm(dKssi - dKssi_GPML);
disp(['      norm(dKssi - dKssi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end
disp('   testing output scale derivative (i = 2)');
dKssi = covSEiso(hyp, x, 'diag', 2);
disp('      dKssi = covSEiso(hyp, x, ''diag'', 2)');
disp('   testing for consistency with GPML implementation');
dKssi_GPML = GPMLcovSEiso(hyp, x, 'diag', 2);
disp('      dKssi_GPML = GPMLcovSEiso(hyp, x, ''diag'', 2)');
error = norm(dKssi - dKssi_GPML);
disp(['      norm(dKssi - dKssi_GPML) = ' num2str(error)]);
if (error < tolerance)
  disp('   PASS.');
else
  failed = true;
  disp('   FAIL!');
end

if (failed)
  disp('not all tests passed!');
else
  disp('all tests passed.');
end