function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE, Normalizes the data mean 0 and std 1.
%   X_norm--> Normalized Data. MAtrix [M,N]
%   mu--> Mean of each variable. Vector [1,N]
%   sigma--> Standard deviation for each variable. Vector[1,N]
   
    mu = mean(X);
    sigma = std(X);
    X_norm = bsxfun(@minus, X, mu);
    X_norm = bsxfun(@rdivide, X_norm, sigma);

end