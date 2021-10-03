function [T] = basicStats(X)
%basicStats returns a table with tha basic statistics.
%   X must be a matrix



att = {'sumary', 'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'};
mu = mean(X);
median1 = median(X);
std_dev = std(X);
count = [];
maxV = max(X);
minV = min(X);
q25 = quantile(X, 0.25);
q50 = quantile(X, 0.50);
q75 = quantile(X, 0.75);

[~, N] = size(X);
for i = 1:N
    count = [count length(X(:,i))];
end

sumar = [count; mu; median1; std_dev; maxV; minV; q25; q50; q75];
T1 = array2table(sumar);
S =  ["count"; "mu"; "median"; "std_dev"; "max"; "min"; "25%"; "50%"; "75%"];
T2 = table(S);

T = [T2, T1];
T.Properties.VariableNames = att;
end