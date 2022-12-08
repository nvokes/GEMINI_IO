clearvars
clc

tblMDA = readtable('sorted_MDA_train_ICI_Mono.xlsx');
tblMDA.Age(tblMDA.Age <= 65) = 0;
tblMDA.Age(tblMDA.Age > 65) = 1;
dataMDA = removevars(tblMDA,{'Var1','Single_Combine','Line_of_IO_conden', ...
    'PFS','PFS_Status','OS','OS_Status'});

nameSmoker = ["Never Smoker","Former Smoker","Current Smoker"];
MDASmoker = array2table(dummyvar(categorical(dataMDA.Tobacco_Use)),"VariableNames",nameSmoker);

namePDL1 = ["PDL1 (Low)","PDL1 (intermediate)","PDL1 (high)"];
MDAPDL1 = array2table(dummyvar(categorical(dataMDA.PD_L1_expression)),"VariableNames",namePDL1);

namePathology = ["Pathology (Adenocarcinoma)","Pathology (Squamous)","Pathology (Adenosquamous/NOS)"];
MDAPathology = array2table(dummyvar(categorical(dataMDA.Pathology)),"VariableNames",namePathology);

dataMDA.Tobacco_Use = [];
dataMDA.PD_L1_expression = [];
dataMDA.Pathology = [];

dataMDA = [dataMDA,MDASmoker,MDAPDL1,MDAPathology];

tblMDAExt = readtable('sorted_MDA_val_ICI_Mono.xlsx');
tblMDAExt.Age(tblMDAExt.Age <= 65) = 0;
tblMDAExt.Age(tblMDAExt.Age > 65) = 1;
dataMDAExt = removevars(tblMDAExt,{'Var1','Single_Combine','Line_of_IO_conden', ...
    'PFS','PFS_Status','OS','OS_Status'});

MDAExtSmoker = array2table(dummyvar(categorical(dataMDAExt.Tobacco_Use)),"VariableNames",nameSmoker);

MDAExtPDL1 = array2table(dummyvar(categorical(dataMDAExt.PD_L1_expression)),"VariableNames",namePDL1);

MDAExtPathology = array2table(dummyvar(categorical(dataMDAExt.Pathology)),"VariableNames",namePathology);

dataMDAExt.Tobacco_Use = [];
dataMDAExt.PD_L1_expression = [];
dataMDAExt.Pathology = [];

dataMDAExt = [dataMDAExt,MDAExtSmoker,MDAExtPDL1,MDAExtPathology];

tblMayo = readtable('sorted_Mayo_val_ICI_Mono.xlsx');
dataMayo = removevars(tblMayo,{'Var1','Single_Combine','Line_of_IO_conden', ...
    'PFS','PFS_Status','OS','OS_Status'});

MayoSmoker = array2table(dummyvar(categorical(dataMayo.Tobacco_Use)),"VariableNames",nameSmoker);

MayoPDL1 = array2table(dummyvar(categorical(dataMayo.PD_L1_expression)),"VariableNames",namePDL1);

MayoPathology = array2table(dummyvar(categorical(dataMayo.Pathology)),"VariableNames",namePathology);

dataMayo.Tobacco_Use = [];
dataMayo.PD_L1_expression = [];
dataMayo.Pathology = [];

dataMayo = [dataMayo,MayoSmoker,MayoPDL1,MayoPathology];

for i = 1:width(dataMDA)
    if nnz(table2array(dataMDA(:,i)))<4
        discard{i} = i;
    else
        discard{i} = 0;
    end
end
discard = nonzeros(cell2mat(discard));
dataMDA(:,discard) = [];

respVarMDA = logical(dataMDA.prog_3_mo);
respVarMDAExt = logical(dataMDAExt.prog_3_mo);
respVarMayo = logical(dataMayo.prog_3_mo);

dataMDA.prog_3_mo = [];
dataMDAExt.prog_3_mo = [];
dataMayo.prog_3_mo = [];

tmpvarNamesMDA = dataMDA.Properties.VariableNames;
tmpvarNamesMDAExt = dataMDAExt.Properties.VariableNames;
tmpvarNamesMayo = dataMayo.Properties.VariableNames;

[Lia,Locb] = ismember(tmpvarNamesMDA,tmpvarNamesMDAExt);

varNamesMDA = tmpvarNamesMDA(Lia);
varNamesMDAExt = tmpvarNamesMDAExt(nonzeros(Locb));
varNamesMayo = tmpvarNamesMayo(nonzeros(Locb));

dataMDA = table2array(dataMDA(:,varNamesMDA));
dataMDAExt = table2array(dataMDAExt(:,varNamesMDAExt));
dataMayo = table2array(dataMayo(:,varNamesMayo));

classNames = unique(respVarMDAExt);

%% feature ranking using chi-square tests
[idxChi,scoresChi] = fscchi2(dataMDA,respVarMDA);
idxInf = find(isinf(scoresChi));
chiFeas = cell2table(varNamesMDA(idxChi)');
scores = scoresChi(idxChi);
FeaRank = cell2table(varNamesMDA(idxChi)');
FeaRank.scores = (scores/sum(scores))';
% writetable(FeaRank,'featureRanking_ICI_Mono.xlsx')

% feature importance bar plot
[x,index] = sort(FeaRank.scores(1:10));
y = strrep(FeaRank.Var1(1:10),'_','\_');
figure();
b = barh(x);
xlabel('feature importance score')
yticklabels(y(index))

%% fit a logistic regression model
rng('default')
tempLR = cell(length(idxChi),6);
AUCLog = zeros(length(idxChi),3);
for lr = 1:length(idxChi)
    disp([num2str(lr), '/', num2str(length(idxChi))]);
    CVmdlLog = fitclinear(dataMDA(:,idxChi(1:lr)),respVarMDA,'ObservationsIn','rows','KFold',10,...
        'Learner','logistic','Solver','sparsa','Regularization','lasso',...
        'GradientTolerance',1e-8);
    mdlLogMM = fitclinear(dataMDA(:,idxChi(1:lr)),respVarMDA,'ObservationsIn','rows',...
        'Learner','logistic','Solver','sparsa','Regularization','lasso',...
        'GradientTolerance',1e-8);

    [~,ScoresLog] = kfoldPredict(CVmdlLog);
    [~,ScoresLogMM] = predict(mdlLogMM,dataMDAExt(:,idxChi(1:lr)));
    [~,ScoresLogMY] = predict(mdlLogMM,dataMayo(:,idxChi(1:lr)));

    [Xlr,Ylr,Tlr,AUClr] = perfcurve(respVarMDA,ScoresLog(:,CVmdlLog.ClassNames),'true');
    [XlrMM,YlrMM,TlrMM,AUClrMM] = perfcurve(respVarMDAExt,ScoresLogMM(:,mdlLogMM.ClassNames),'true');
    [XlrMY,YlrMY,TlrMY,AUClrMY] = perfcurve(respVarMayo,ScoresLogMY(:,mdlLogMM.ClassNames),'true');

    AUCLog(lr,1) = AUClr;
    AUCLog(lr,2) = AUClrMM;
    AUCLog(lr,3) = AUClrMY;

    tempLR{lr,1} = Xlr;
    tempLR{lr,2} = Ylr;
    tempLR{lr,3} = XlrMM;
    tempLR{lr,4} = YlrMM;
    tempLR{lr,5} = XlrMY;
    tempLR{lr,6} = YlrMY;
end

%% fit a support vector machine
tempSVM = cell(length(idxChi),6);
AUCSVM = zeros(length(idxChi),3);
for sv = 1:length(idxChi)
    disp([num2str(sv), '/', num2str(length(idxChi))]);
    mdlSVM = fitcsvm(dataMDA(:,idxChi(1:sv)),respVarMDA);
    CVmdlSVM = crossval(mdlSVM);

    [~,Scoressvm] = kfoldPredict(CVmdlSVM);
    [~,ScoressvmMM] = predict(mdlSVM,dataMDAExt(:,idxChi(1:sv)));
    [~,ScoressvmMY] = predict(mdlSVM,dataMayo(:,idxChi(1:sv)));

    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(respVarMDA,Scoressvm(:,CVmdlSVM.ClassNames),'true');
    [XsvmMM,YsvmMM,TsvmMM,AUCsvmMM] = perfcurve(respVarMDAExt,ScoressvmMM(:,mdlSVM.ClassNames),'true');
    [XsvmMY,YsvmMY,TsvmMY,AUCsvmMY] = perfcurve(respVarMayo,ScoressvmMY(:,mdlSVM.ClassNames),'true');

    AUCSVM(sv,1) = AUCsvm;
    AUCSVM(sv,2) = AUCsvmMM;
    AUCSVM(sv,3) = AUCsvmMY;

    tempSVM{sv,1} = Xsvm;
    tempSVM{sv,2} = Ysvm;
    tempSVM{sv,3} = XsvmMM;
    tempSVM{sv,4} = YsvmMM;
    tempSVM{sv,5} = XsvmMY;
    tempSVM{sv,6} = YsvmMY;
end

%% fit a random forest model
tempRF = cell(length(idxChi),6);
AUCRF = zeros(length(idxChi),3);
% t = templateTree('MaxNumSplits',5);
for rf = 1:length(idxChi)
    disp([num2str(rf), '/', num2str(length(idxChi))]);
    mdlRF = fitcensemble(dataMDA(:,idxChi(1:rf)),respVarMDA);
    CVmdlRF = crossval(mdlRF);

    [~,Scoresrf] = kfoldPredict(CVmdlRF);
    [~,ScoresrfMM] = predict(mdlRF,dataMDAExt(:,idxChi(1:rf)));
    [~,ScoresrfMY] = predict(mdlRF,dataMayo(:,idxChi(1:rf)));


    [Xrf,Yrf,Trf,AUCrf] = perfcurve(respVarMDA,Scoresrf(:,CVmdlRF.ClassNames),'true');
    [XrfMM,YrfMM,TrfMM,AUCrfMM] = perfcurve(respVarMDAExt,ScoresrfMM(:,mdlRF.ClassNames),'true');
    [XrfMY,YrfMY,TrfMY,AUCrfMY] = perfcurve(respVarMayo,ScoresrfMY(:,mdlRF.ClassNames),'true');


    AUCRF(rf,1) = AUCrf;
    AUCRF(rf,2) = AUCrfMM;
    AUCRF(rf,3) = AUCrfMY;

    tempRF{rf,1} = Xrf;
    tempRF{rf,2} = Yrf;
    tempRF{rf,3} = XrfMM;
    tempRF{rf,4} = YrfMM;
    tempRF{rf,5} = XrfMY;
    tempRF{rf,6} = YrfMY;
end

%% fit a generalized additive model 
tempGAM = cell(length(idxChi),6);
AUCGAM = zeros(length(idxChi),3);
for g = 1:length(idxChi)
    disp([num2str(g), '/', num2str(length(idxChi))]);
    mdlGAM = fitcgam(dataMDA(:,idxChi(1:g)),respVarMDA);
    CVmdlGAM = crossval(mdlGAM);

    [~,Scoresgam] = kfoldPredict(CVmdlGAM);
    [~,ScoresgamMM] = predict(mdlGAM,dataMDAExt(:,idxChi(1:g)));
    [~,ScoresgamMY] = predict(mdlGAM,dataMayo(:,idxChi(1:g)));

    [Xgam,Ygam,Tgam,AUCgam] = perfcurve(respVarMDA,Scoresgam(:,CVmdlGAM.ClassNames),'true');
    [XgamMM,YgamMM,TgamMM,AUCgamMM] = perfcurve(respVarMDAExt,ScoresgamMM(:,mdlGAM.ClassNames),'true');
    [XgamMY,YgamMY,TgamMY,AUCgamMY] = perfcurve(respVarMayo,ScoresgamMY(:,mdlGAM.ClassNames),'true');

    AUCGAM(g,1) = AUCgam;
    AUCGAM(g,2) = AUCgamMM;
    AUCGAM(g,3) = AUCgamMY;

    tempGAM{g,1} = Xgam;
    tempGAM{g,2} = Ygam;
    tempGAM{g,3} = XgamMM;
    tempGAM{g,4} = YgamMM;
    tempGAM{g,5} = XgamMY;
    tempGAM{g,6} = YgamMY;
end

%% plot AUC curves for cross-validated models
[bestAUCLR,Ilr] = max(AUCLog);
[bestAUCSVM,Isv] = max(AUCSVM);
[bestAUCRF,Irf] = max(AUCRF);
[bestAUCgam,Ig] = max(AUCGAM);

figure()
plot(1:length(idxChi),AUCLog(:,1),'LineWidth',2)
hold on
plot(1:length(idxChi),AUCSVM(:,1),'LineWidth',2)
plot(1:length(idxChi),AUCRF(:,1),'LineWidth',2)
plot(1:length(idxChi),AUCGAM(:,1),'LineWidth',2)
ylim([0.2 0.8])
xlim([0 41])
l = legend(sprintf("LR (Best AUC = %g, # vars = %g)",round(bestAUCLR(1),2),Ilr(1)), ...
    sprintf("SVM (Best AUC = %g, # vars = %g)",round(bestAUCSVM(1),2),Isv(1)), ...
    sprintf("RF (Best AUC = %g, # vars = %g)",round(bestAUCRF(1),2),Irf(1)), ...
    sprintf("GAM (Best AUC = %g, # vars = %g)",round(bestAUCgam(1),2),Ig(1)),'Location','Best');
l.FontSize = 12;
legend boxoff    

xlabel('Number of variables'); 
ylabel('AUC');
title('AUC Curves (MDA Training)')
hold off

% plot AUC curves for MDA validation
figure()
plot(1:length(idxChi),AUCLog(:,2),'LineWidth',2)
hold on
plot(1:length(idxChi),AUCSVM(:,2),'LineWidth',2)
plot(1:length(idxChi),AUCRF(:,2),'LineWidth',2)
plot(1:length(idxChi),AUCGAM(:,2),'LineWidth',2)
ylim([0.2 0.9])
xlim([0 41])
ll = legend(sprintf("LR (Best AUC = %g, # vars = %g)",round(bestAUCLR(2),2),Ilr(2)), ...
    sprintf("SVM (Best AUC = %g, # vars = %g)",round(bestAUCSVM(2),2),Isv(2)), ...
    sprintf("RF (Best AUC = %g, # vars = %g)",round(bestAUCRF(2),2),Irf(2)), ...
    sprintf("GAM (Best AUC = %g, # vars = %g)",round(bestAUCgam(2),2),Ig(2)),'Location','Best');
ll.FontSize = 12;
legend boxoff    

xlabel('Number of variables');
ylabel('AUC');
title('AUC Curves (MDA Validation)')
hold off

% plot AUC curves for Mayo validation
figure()
plot(1:length(idxChi),AUCLog(:,3),'LineWidth',2)
hold on
plot(1:length(idxChi),AUCSVM(:,3),'LineWidth',2)
plot(1:length(idxChi),AUCRF(:,3),'LineWidth',2)
plot(1:length(idxChi),AUCGAM(:,3),'LineWidth',2)
ylim([0.2 1])
xlim([0 41])
ll = legend(sprintf("LR (Best AUC = %g, # vars = %g)",round(bestAUCLR(3),2),Ilr(3)), ...
    sprintf("SVM (Best AUC = %g, # vars = %g)",round(bestAUCSVM(3),2),Isv(3)), ...
    sprintf("RF (Best AUC = %g, # vars = %g)",round(bestAUCRF(3),2),Irf(3)), ...
    sprintf("GAM (Best AUC = %g, # vars = %g)",round(bestAUCgam(3),2),Ig(3)),'Location','Best');
ll.FontSize = 12;
legend boxoff    

xlabel('Number of variables');
ylabel('AUC');
title('AUC Curves (Mayo Validation)')
hold off

%% plot ROC curves
% first fit logistic regression for PDL1
PDL1CVMdl = fitclinear(dataMDA(:,matches(varNamesMDA,["PDL1 (Low)","PDL1 (intermediate)","PDL1 (high)"])),...
    respVarMDA,'ObservationsIn','rows','KFold',10,'Learner','logistic','Solver','sparsa','Regularization','lasso',...
    'GradientTolerance',1e-8);
[~,ScoresLogPDL1] = kfoldPredict(PDL1CVMdl);
[X,Y,T,AUC] = perfcurve(respVarMDA,ScoresLogPDL1(:,PDL1CVMdl.ClassNames),'true');

PDL1Mdl = fitclinear(dataMDA(:,matches(varNamesMDA,["PDL1 (Low)","PDL1 (intermediate)","PDL1 (high)"])),...
    respVarMDA,'ObservationsIn','rows','Learner','logistic','Solver','sparsa','Regularization','lasso',...
    'GradientTolerance',1e-8);
[~,ScoresMMPDL1] = predict(PDL1Mdl,dataMDAExt(:,matches(varNamesMDA,["PDL1 (Low)","PDL1 (intermediate)","PDL1 (high)"])));
[~,ScoresMYPDL1] = predict(PDL1Mdl,dataMayo(:,matches(varNamesMDA,["PDL1 (Low)","PDL1 (intermediate)","PDL1 (high)"])));

[XMMPDL1,YMMPDL1,TMMPDL1,AUCMMPDL1] = perfcurve(respVarMDAExt,ScoresMMPDL1(:,PDL1Mdl.ClassNames),'true');
[XMYPDL1,YMYPDL1,TMYPDL1,AUCMYPDL1] = perfcurve(respVarMayo,ScoresMYPDL1(:,PDL1Mdl.ClassNames),'true');

% Compare ROC curves for LR cross-validated model and PDL1 (MDA train)
figure()
Xlog = cell2mat(tempLR(Ilr(1),1));
Ylog = cell2mat(tempLR(Ilr(1),2));
plot(Xlog,Ylog,'Color','#0072BD','LineWidth',2)
hold on
plot(X,Y,'Color','r','LineWidth',2);
xlabel('1-Specificity')
ylabel('Sensitivity')
title('MDA CV model vs MDA PDL1')

legend(sprintf("LR (AUC = %g)",round(bestAUCLR(1),2)),sprintf("PD-L1 (AUC = %g)",round(AUC,2)), ...
    'Location','Best')
legend boxoff    
hold off

% Compare ROC curves for LR model and PDL1 (MDA validation)
figure()
Xlog1 = cell2mat(tempLR(Ilr(2),3));
Ylog1 = cell2mat(tempLR(Ilr(2),4));
plot(Xlog1,Ylog1,'Color','#0072BD','LineWidth',2)
hold on
plot(XMMPDL1,YMMPDL1,'Color','r','LineWidth',2);
xlabel('1-Specificity')
ylabel('Sensitivity')
title('MDA val model vs MDA val PDL1')

legend(sprintf("LR (AUC = %g)",round(bestAUCLR(2),2)),sprintf("PD-L1 (AUC = %g)",round(AUCMMPDL1,2)), ...
    'Location','Best')
legend boxoff    
hold off

% Compare ROC curves for LR model and PDL1 (Mayo validation)
figure()
Xlog2 = cell2mat(tempLR(Ilr(3),5));
Ylog2 = cell2mat(tempLR(Ilr(3),6));
plot(Xlog2,Ylog2,'Color','#0072BD','LineWidth',2)
hold on
plot(XMYPDL1,YMYPDL1,'Color','r','LineWidth',2);
xlabel('1-Specificity')
ylabel('Sensitivity')
title('Mayo val model vs Mayo val PDL1')

legend(sprintf("LR (AUC = %g)",round(bestAUCLR(3),2)),sprintf("PD-L1 (AUC = %g)",round(AUCMYPDL1,2)), ...
    'Location','Best')
legend boxoff    
hold off
