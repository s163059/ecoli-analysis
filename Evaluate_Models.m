%% Restart
clc; clear; close all;
%% Data
load 'Regression2FoldValues4.mat'
K = 5;
TestSize = 1;
EANN = Error_test_ann';
ELR = Error_test_rlr';
EBL = Error_test_nofeatures';
%% Determine if classifiers are significantly different.
% The function ttest computes credibility interval 
figure();
hold on;
title('Boxplot of model generalization error');
boxplot([EANN./TestSize; ELR./TestSize; EBL./TestSize]'*100, ...
    'labels', {'Artificial Neural Network', 'Linear Regression Model', 'Baseline Model'});
ylabel('Error rate (%)');
hold off;
%% For Next Plot
zk = [(EANN-ELR)./TestSize; (EANN-EBL)./TestSize; (ELR-EBL)./TestSize];
zb = mean(zk')';
sigmaEst = sqrt( mean( (zk-zb)'.^2) / (K-1));
nu = K-1;
alpha = 0.05;
%% PLOT
k = [0:.005:1];
m = sign(k);

xANN = zb(1) + sigmaEst(1) * icdf('T', k, nu);
xLR = zb(2) + sigmaEst(2) * icdf('T', k, nu);
xBL = zb(3) + sigmaEst(3) * icdf('T', k, nu);



% 5% confidence intervals:
XLANN = zb(1) + sigmaEst(1) * tinv([alpha/2, 1-alpha/2], nu);
XLLR = zb(2) + sigmaEst(2) * tinv([alpha/2, 1-alpha/2], nu);
XLBL = zb(3) + sigmaEst(3) * tinv([alpha/2, 1-alpha/2], nu);

%y-values
k = [-1:.01:1];
y = tcdf(k,nu);

%figure()
%plot(k,y)

figure()
hold on
title({'t cumulative distribution', 'for each model with credibility intervals', '\alpha = 0.05'});
%ANN
orange = [0.9290 0.6940 0.1250];
plot(xANN,y, 'Color', orange, 'LineWidth', 1)
line([XLANN(1), XLANN(1)], ylim, 'Color', orange);
line([XLANN(2), XLANN(2)], ylim, 'Color', orange);

%LR
blue = [0 0.4470 0.7410];
plot(xLR,y,'Color', blue, 'LineWidth', 1)
line([XLLR(1), XLLR(1)], ylim,'Color', blue);
line([XLLR(2), XLLR(2)], ylim,'Color', blue);

%BL
red = [0.8500 0.3250 0.0980];
plot(xBL,y,'Color', red, 'LineWidth', 1)
line([XLBL(1), XLBL(1)], ylim,'Color',red);
line([XLBL(2), XLBL(2)], ylim,'Color',red);

%ylim([0.22 0.38])
%xlim([-800 800])
legend({'\DeltaE1 (ANN vs Linear Regression Model)', '0.05%', '0.95%','\DeltaE2 (ANN vs Baseline Model)', '0.05%', '0.95%','\DeltaE3 (Linear Regression Model vs Baseline Model)', '0.05%', '0.95%'})
ylabel('Probability Density');
xlabel('\DeltaError Rate');
grid minor;


hold off
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Data

K = 5;
TestSize = 1;
EANN = [0.06547619;0.08333334;0.07738096;0.09523810;0.05952381;0.08333333;0.07142857;0.05357143;0.04166667;0.05952381]';
ELR = [0.26470588;0.23529412;0.212121212;0.18181818;0.18181818;0.27272727;0.30303030;0.21212121;0.24242424;0.45454545]';
EBL = [0.55882353;0.55882353;0.54545455;0.57575758;0.57575758;0.57575758;0.57575758;0.57575758;0.57575758;0.57575758]';
%% Determine if classifiers are significantly different.
% The function ttest computes credibility interval 
figure();
hold on;
title('Boxplot of model generalization error');
boxplot([EANN./TestSize; ELR./TestSize; EBL./TestSize]'*100, ...
    'labels', {'Artificial Neural Network', 'Linear Regression Model', 'Baseline Model'});
ylabel('Error rate (%)');
hold off;
%% For Next Plot
zk = [(EANN-ELR)./TestSize; (EANN-EBL)./TestSize; (ELR-EBL)./TestSize];
zb = mean(zk')';
sigmaEst = sqrt( mean( (zk-zb)'.^2) / (K-1));
nu = K-1;
alpha = 0.05;
%% PLOT
k = [0:.005:1];
m = sign(k);

xANN = zb(1) + sigmaEst(1) * icdf('T', k, nu);
xLR = zb(2) + sigmaEst(2) * icdf('T', k, nu);
xBL = zb(3) + sigmaEst(3) * icdf('T', k, nu);



% 5% confidence intervals:
XLANN = zb(1) + sigmaEst(1) * tinv([alpha/2, 1-alpha/2], nu);
XLLR = zb(2) + sigmaEst(2) * tinv([alpha/2, 1-alpha/2], nu);
XLBL = zb(3) + sigmaEst(3) * tinv([alpha/2, 1-alpha/2], nu);

%y-values
k = [-1:.01:1];
y = tpdf(k,nu);

%figure()
%plot(k,y)

figure(3)
hold on
title({'t cumulative distribution', 'for each model with credibility intervals', '\alpha = 0.05'});
%ANN
orange = [0.9290 0.6940 0.1250];
plot(xANN,y, 'Color', orange, 'LineWidth', 1)
line([XLANN(1), XLANN(1)], ylim, 'Color', orange);
line([XLANN(2), XLANN(2)], ylim, 'Color', orange);

%LR
blue = [0 0.4470 0.7410];
plot(xLR,y,'Color', blue, 'LineWidth', 1)
line([XLLR(1), XLLR(1)], ylim,'Color', blue);
line([XLLR(2), XLLR(2)], ylim,'Color', blue);

%BL
red = [0.8500 0.3250 0.0980];
plot(xBL,y,'Color', red, 'LineWidth', 1)
line([XLBL(1), XLBL(1)], ylim,'Color',red);
line([XLBL(2), XLBL(2)], ylim,'Color',red);

ylim([0.2168 0.38])
xlim([-0.55 0.0001])
legend({'\DeltaE1 (ANN vs Multinomial Regression Model)', '0.05%', '0.95%','\DeltaE2 (ANN vs Baseline Model)', '0.05%', '0.95%','\DeltaE3 (Multinomial Regression Model vs Baseline Model)', '0.05%', '0.95%'})
ylabel('Probability Density');
xlabel('\DeltaError Rate');
grid minor;

hold off

