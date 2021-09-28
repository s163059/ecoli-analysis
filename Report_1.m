clc; clear all;
%% Read the data.

X = readtable('ecoli.csv');

% Select attribute names
att = {'prot_name', 'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'cat'};
X.Properties.VariableNames = att;

X(:,'prot_name');
% If we want to use more than one var use brakets

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

cat = categorical({'cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'});

% Substract the class labels from the matrix
classLabels = table2cell(X(:,9));
% Fiter the uniques values
classNames = unique(classLabels);
% Extract class labels that match the class names
[~,y] = ismember(classLabels, classNames);
% Using '~' ignores an output. Try writing 'help ismember'. Here, we use
% the output that the doc calls LOCB to determine to which class name each
% class label in classLabels corresponds. Since classLabels(75) is an
% 'Iris-versicolor', we could call:
%[~, b] = ismember(classLabels(75), classNames)
% to see that classLabels(1) corresponds to b=2, and therefore the second
% class name in classNames.
% Since we want to assign numerical values to the classes starting from a
% zero and not a one, we subtract one to the get final y:
y = y-1;
length(y);
% Lastly, we determine the number of attributes M, the number of
% observations N and the number of classes C:
[M, N] = size(X);
C = length(classNames);

% Create a matrix 
ecoli_temp = table2array(X(:, {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'}));
ecoli = [ecoli_temp, y];




% histogram(y, 'labels', classNames);



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




