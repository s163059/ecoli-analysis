clc; clear all;
%% Read the data.

X = readtable('ecoli.csv');

% Select attribute names
att = {'prot_name', 'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'cat'};
X.Properties.VariableNames = att;

X(:,'prot_name');
% If we want to use more than one var use brakets


%% Label encoding
% Substract the column class from the matrix. The column we wish to predict
classLabels = table2cell(X(:,9));
% Fiter the uniques values
classNames = unique(classLabels);
% Create an encode vector of the different categories
%{
Transform the different categories to numbers 
  cp  (cytoplasm)                                      0
  im  (inner membrane without signal sequence)         1               
  pp  (perisplasm)                                     2
  imU (inner membrane, uncleavable signal sequence)    3
  om  (outer membrane)                                 4
  omL (outer membrane lipoprotein)                     5
  imL (inner membrane lipoprotein)                     6
  imS (inner membrane, cleavable signal sequence)      7
%}
[~,y] = ismember(classLabels, classNames);
% Substract 1 to the vector so it's starts from 0.
y = y-1;
y_len = length(y);
% Lastly, we determine the number of attributes M, the number of
% observations N and the number of classes C:
[M, N] = size(X);
C = length(classNames);

% Create a matrix with the variables that are really useful. So we can
% create an easy further visual analysis.
ecoli_temp = table2array(X(:, {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'}));
% Append the y encoded vector to the matrix. It's not really essential
ecoli = [ecoli_temp, y];
size(ecoli)
% Create a categoriacal vector with the attributes from the data. So we can
% use them as labels later (not sure if it's necessesary)
cat = categorical({'cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'});

% Check if we can get some information with histograms.
figure()
subplot(3,2,1)
histogram(ecoli(:,1), 600);

subplot(3,2,2)
histogram(ecoli(:,1), 600);

subplot(3,2,3)
histogram(ecoli(:,1), 600);

subplot(3,2,4)
histogram(ecoli(:,1), 600);

subplot(3,2,5)
histogram(ecoli(:,1), 600);

subplot(3,2,6)
histogram(ecoli(:,1), 600);



figure()
heatmap(ecoli)

F = ecoli'*ecoli/mean(ecoli);
G = corr(ecoli);
F == G




% Box plot to analyse the importance of the first variable
figure()
boxplot(ecoli(:,1), y, 'labels', classNames)
xlabel('Categories')
ylabel('Score of mgc')
title('Quantity of mgc in each category')
% If we only focused on the mean, could diferenciate two different
% categories, one that's above 0.5 and and another that's below 0.5.

% We could also say that will have to focus on both goups to see how we
% should separate them


%We can plot something like in page 122 of the book




