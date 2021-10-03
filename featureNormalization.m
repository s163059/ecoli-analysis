function [X_norm] = featureNormalization(X)
% FEATURENORMALIZATION Normalized the data 

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);


end


%{
% correlation matrix 
E_correlation = E_norm'*E_norm/(length(E_norm(:,1))-1);
figure()
heatmap(E_correlation)
% Same result, only to check

E_corr = corr(ecoli);
figure()
heatmap(E_corr)
% heatmap of covariance 
%}