function [T] = PCAComponents(eigval)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

prop = eigval/sum(eigval);
cumulative = cumsum(prop);
diff = zeros(length(eigval), 1);
for i= 1:length(eigval)
    if i < length(eigval)
        diff(i) =  (eigval(i) - eigval(i+1));
        
    else 
        diff(i) = 0;
    end
end


index = (1:length(eigval))';

summary = [eigval, diff, prop, cumulative];
summary = [index, summary];

T = array2table(summary);

att = {'PCA', 'Eigenvalues', 'Difference', 'Proportion', 'Cumulative'};
T.Properties.VariableNames = att;



end

