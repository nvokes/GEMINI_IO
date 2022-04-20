clearvars
clc

tbl = readtable('Data_reduced_PDL1_NA_removed_ICI_Chemo.xlsx');
data = removevars(tbl,{'S_N','Row_names','Single_Combine','Line_of_IO_conden','PFS','PFS_Status','OS','OS_Status'});
for i = 1:width(data)
    if nnz(table2array(data(:,i)))<4
        discard{i} = i;
    else
        discard{i} = 0;
    end
end
discard = nonzeros(cell2mat(discard));
data(:,discard) = [];

respVar = logical(data.prog_3_mo);
data.prog_3_mo = [];
varNames = data.Properties.VariableNames;
data = table2array(data);

%% feature ranking using chi-square tests
[idxChi,scoresChi] = fscchi2(data,respVar);
idxInf = find(isinf(scoresChi));
chiFeas = cell2table(varNames(idxChi)');
scores = scoresChi(idxChi);
FeaRank = cell2table(varNames(idxChi)');
FeaRank.scores = scores';
% writetable(FeaRank,'featureRanking_ICI_Chemo.xlsx')

%% fit a logistic regression model
rng('default')
tempLR = cell(length(idxChi),2);
AUCLog = zeros(length(idxChi),1);
for lr = 1:length(idxChi)
    disp([num2str(lr), '/', num2str(length(idxChi))]);
    mdlLog = fitclinear(data(:,idxChi(1:lr)),respVar,'ObservationsIn','rows','KFold',10,...
    'Learner','logistic','Solver','sparsa','Regularization','lasso',...
    'GradientTolerance',1e-8);
    [~,score_log] = kfoldPredict(mdlLog);
    [Xlr,Ylr,Tlr,AUClr] = perfcurve(respVar,score_log(:,mdlLog.ClassNames),'true');
    AUCLog(lr) = AUClr;
    tempLR{lr,1} = Xlr;
    tempLR{lr,2} = Ylr;
end

%% fit a support vector machine
tempSVM = cell(length(idxChi),2);
AUCSVM = zeros(length(idxChi),1);
for sv = 1:length(idxChi)
    disp([num2str(sv), '/', num2str(length(idxChi))]);
    mdlSVM = fitcsvm(data(:,idxChi(1:sv)),respVar, 'Crossval','on');
    CVmdlSVM = fitSVMPosterior(mdlSVM);
    [~,score_svm] = kfoldPredict(CVmdlSVM);
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(respVar,score_svm(:,CVmdlSVM.ClassNames),'true');
    AUCSVM(sv) = AUCsvm;
    tempSVM{sv,1} = Xsvm;
    tempSVM{sv,2} = Ysvm;
end

%% fit a random forest model
tempRF = cell(length(idxChi),2);
AUCRF = zeros(length(idxChi),1);
% t = templateTree('MaxNumSplits',5);
for rf = 1:length(idxChi)
    disp([num2str(rf), '/', num2str(length(idxChi))]);
    mdlRF = fitcensemble(data(:,idxChi(1:rf)),respVar, 'Crossval','on');
    [~,score_rf] = kfoldPredict(mdlRF);
    [Xrf,Yrf,Trf,AUCrf] = perfcurve(respVar,score_rf(:,mdlRF.ClassNames),'true');
    AUCRF(rf) = AUCrf;
    tempRF{rf,1} = Xrf;
    tempRF{rf,2} = Yrf;
end

%% fit a generalized additive model 
tempGAM = cell(length(idxChi),2);
AUCGAM = zeros(length(idxChi),1);
for g = 1:length(idxChi)
    disp([num2str(g), '/', num2str(length(idxChi))]);
    mdlGAM = fitcgam(data(:,idxChi(1:g)),respVar, 'Crossval','on');
    [~,score_gam] = kfoldPredict(mdlGAM);
    [Xgam,Ygam,Tgam,AUCgam] = perfcurve(respVar,score_gam(:,mdlGAM.ClassNames),'true');
    AUCGAM(g) = AUCgam;
    tempGAM{g,1} = Xgam;
    tempGAM{g,2} = Ygam;
end

%% plot AUC curves
plot(1:length(idxChi),AUCLog,'LineWidth',2)
hold on
plot(1:length(idxChi),AUCSVM,'LineWidth',2)
plot(1:length(idxChi),AUCRF,'LineWidth',2)
plot(1:length(idxChi),AUCGAM,'LineWidth',2)
ylim([0.2 0.8])
xlim([0 46])
l = legend('LR','SVM','RF','GAM','Location','Best');
l.FontSize = 12;
legend boxoff    

xlabel('Number of variables'); 
ylabel('AUC');
title('AUC Curves for LR, SVM, RF, and GAM')
hold off

%% plot ROC curves for best models
[bestAUCLR,Ilr] = max(AUCLog);
[bestAUCSVM,Isv] = max(AUCSVM);
[bestAUCRF,Irf] = max(AUCRF);
[bestAUCgam,Ig] = max(AUCGAM);

figure()
plot(cell2mat(tempLR(Ilr,1)),cell2mat(tempLR(Ilr,2)),'r','LineWidth',2)
hold on
plot(cell2mat(tempSVM(Isv,1)),cell2mat(tempSVM(Isv,2)),'g','LineWidth',2)
plot(cell2mat(tempRF(Irf,1)),cell2mat(tempRF(Irf,2)),'b','LineWidth',2)
plot(cell2mat(tempGAM(Ig,1)),cell2mat(tempGAM(Ig,2)),'k','LineWidth',2)

l = legend('LR','SVM','RF','GAM','Location','Best');
legend boxoff    
l.FontSize = 12;
title('ROC Curves for LR, SVM, RF, and GAM')
xlabel('1-Specificity')
ylabel('Sensitivity')
