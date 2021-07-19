%                                               |
% EEG-Based Brain-Operated Typewriting Machine  |
% M. Amirsardari - A. H. Mobasheri              |
% Summer 1400/2021                              |
% Part5: Learning and Prediction                |
%_______________________________________________|

% Part5_Dataset:
clear; clc; close all;

% first of all load data and seperate train and test 
datap(1) = load("dataset\SubjectData1.mat");
datap(2) = load("dataset\SubjectData2.mat");
datap(3) = load("dataset\SubjectData3.mat");
%datap4 = load("dataset\SubjectData4.mat");
datap(5) = load("dataset\SubjectData5.mat");
datap(6) = load("dataset\SubjectData6.mat");
datap(7) = load("dataset\SubjectData7.mat");
datap(8) = load("dataset\SubjectData8.mat");
datap(9) = load("dataset\SubjectData9.mat");

%%
% Filtering:
clc;

Fs = 1/(datap(1).train(1,6) - datap(1).train(1,5));
temp_test = struct;
temp_train = struct;


tic
for i = [1, 2, 3, 5, 6, 7, 8, 9]
    for ch = [1, 10, 11]
        temp_train(i).ch(ch,:) = datap(i).train(ch,:);
    end
    for ch = 2:9
        temp_train(i).ch(ch,:) = bandpass(datap(i).train(ch,:), [0.5, 40], Fs);  
    end
    for ch = 2:9
        temp_test(i).ch(ch,:) = bandpass(datap(i).test(ch,:), [0.5, 40], Fs);
    end
    temp_test(i).ch(1,:) = datap(i).test(1,:);
    temp_test(i).ch(10,:) = datap(i).test(10,:);
end

toc % 113 seconds

%%
% Down Sampling:
clc; close all;

L = 3;

test = struct;
train = struct;

tic
for i = [1, 2, 3, 5, 6, 7, 8, 9]
    for ch = 1:11
        train(i).ch(ch,:) = downSampler(temp_train(i).ch(ch,:), L);  
    end
    for ch = 1:10
        test(i).ch(ch,:) = downSampler(temp_test(i).ch(ch,:), L);
    end
end
toc 


%%
% Q10_epoching:
clc; close all;

train_StimuliOnset = struct;
test_StimuliOnset = struct;

targets = struct;

Y = struct;


for i = [1, 2, 3, 5, 6, 7, 8, 9]
     train_StimuliOnset(i).ch = [];
     test_StimuliOnset(i).ch = [];
     targets(i).ch = [];
end

tic
y = struct;
t = struct; 
o = struct;
for i = [1, 2, 3, 5, 6, 7, 8, 9]
    N = length(train(i).ch(10,:));
     k = 1;
    for n = 2:N
        if((train(i).ch(10,n-1) ~= train(i).ch(10,n)) && (train(i).ch(10,n) ~= 0))      
            train_StimuliOnset(i).ch = [train_StimuliOnset(i).ch, n];
            y(i).n(k) = train(i).ch(11,n);
            o(i).n(k) = train(i).ch(10,n);
            k = k + 1;
        end  
    end
    
    N = length(test(i).ch(10,:));
    k = 1;
    for n = 2:N
        if((test(i).ch(10,n-1) ~= test(i).ch(10,n)) && (test(i).ch(10,n) ~= 0) )        
            test_StimuliOnset(i).ch = [test_StimuliOnset(i).ch, n];
            t(i).n(k) = test(i).ch(10,n);
            k = k + 1;
        end   
    end
end
toc % 0.54 sec

%%
clc;

tic
for i = [1, 2, 3, 5, 6, 7, 8, 9]
    epochedTrain(i).ch = epoching(train(i).ch(:,:),0.2,0.8,train_StimuliOnset(i).ch(:));
    epochedTest(i).ch = epoching(test(i).ch(:,:),0.2,0.8,test_StimuliOnset(i).ch(:));
end
toc % 0.37 sec

%%
save('epochedTrain.mat','epochedTrain');
save('epochedTest.mat','epochedTest');

%% implementing machine learning 

clc;
%%% subj 1
m = struct;
yfit = struct;
predTrain = struct;
c = [0,32;0.5,0];
tic 
for k = [1,2,3,5,6,7,8,9]
    
    X = epochedTrain(k).ch;
    Y = transpose(y(k).n);

    X = reshape(X,[size(X,1), size(X,2)*size(X,3)]);

    m(k).model = fitcsvm(X,Y,'cost',c);

    yfit(k).yfit = predict(m(k).model,X);
    
    x = transpose(transpose((o(k).n)) .* (yfit(k).yfit));
    x(x==0) = [];
    predTrain(k).prediction = x;
    
 
   errorSVM(k) = loss(m(k).model,X,Y);

end
toc
predictionTrain = struct;
for k = [1,2,3,5,6,7,8,9]
    if k==1 || k==2 
        predictionTrain(k).num = unique(predTrain(k).prediction,'stable')
    else
  
        x = predTrain(k).prediction;
        x(diff(x) == 0)= [];
        predictionTrain(k).num = x;
              
    end
end

pred2word(predictionTrain(1).num)
pred2word(predictionTrain(2).num)

%%
%%% 55 seconds
c = [0,16;0.5,0];
predTest = struct;
for k = [1,2,3,5,6,7,8,9]
    
    X = epochedTest(k).ch;
    X = reshape(X,[size(X,1), size(X,2)*size(X,3)]);
    
    yhat(k).yhat = predict(m(k).model,X);

    
    
    x = transpose(transpose((t(k).n)) .* (yhat(k).yhat));
    x(x==0) = [];
    predTest(k).prediction = x;
    

end

predictionTest = struct;
for k = [1,2,3,5,6,7,8,9]
    if k==1 || k==2 
        predictionTest(k).num = unique(predTest(k).prediction,'stable')
    else
  
        x = predTest(k).prediction;
        x(diff(x) == 0)= [];
        predictionTest(k).num = x;
              
    end
end






%%
clc;
%%% subj 1
m = struct;
for k = [1,2,3,5,6,7,8,9]
    
    X = epochedTrain(k).ch;
    Y = transpose(y(k).n);

    X = reshape(X,[size(X,1), size(X,2)*size(X,3)]);

    m(k).model = fitcdiscr(X,Y,'cost',c);

    yfit = predict(m(k).model,X);

   errorLDA(k) = loss(m(k).model,X,Y);

end


%%

% train1 = datap1.train;
% train2 = datap2.train;
% train3 = datap3.train;
% %%train4 = datap4.train;
% train5 = datap5.train;
% train6 = datap6.train;
% train7 = datap7.train;
% train8 = datap8.train;
% train9 = datap9.train;
% 
% datap1.train(10,:).* datap1.train(11,:);
% 
% test1 = datap(1).test;
% test2 = datap(2).test;
% test3 = datap(3).test;
% %%test4 = datap4.test;
% test5 = datap(5).test;
% test6 = datap(6).test;
% test7 = datap(7).test;
% test8 = datap(8).test;
% test9 = datap(9).test;

a = struct;

for i = [1, 2, 3, 5, 6, 7, 8, 9]
   % a(i) = unique(train(i).ch(10,:));
end

sc1 = unique(train(1).ch(10,:).*train(1).ch(11,:),'stable');
sc2 = unique(train(2).ch(10,:).*train(2).ch(11,:),'stable');

%%


rc1 = train6(10,:).*train6(11,:);
rc2 = train7(10,:).*train7(11,:);

rc2 = train8(10,:).*train8(11,:);
rc3 = train9(10,:).*train9(11,:);
rc1 = rc1(diff([0 rc1])~=0);
rc1(rc1==0)=[];


fg = indexExtraction(datap2);

%%
%%% IndexExtraction
function out = indexExtraction(data)

        %%% args: 
        %%%            struct
        %%%          
        %%% output:    
        %%%            another struct with timing information
       
        out = data;
        
        out.target = find(data.train(11,:));
        
        out.time_test = find(data.test(10,:));
        
        trains = data.train(10,:);
        zeros = find(trains == 0);
        tars = data.train(11,:);
        tars(zeros) = 2;
        
        
        out.non_target = find(tars == 0);
end


function out = downSampler(Signal, L)
    len = length(Signal);
    N = floor(len/L);
    out = zeros(N,1);
    
    for i=1:N
             out(i) = Signal(L*i);  
    end
end


function epoched = epoching(InputSignal,BackwardSamples,ForwardSamples,StimuliOnset)
    Fs = 3/(InputSignal(1,6)-InputSignal(1,3));
    
    BckIdx = floor(BackwardSamples*Fs);
    ForIdx = floor(ForwardSamples*Fs);
    N = length(StimuliOnset);
    
    epoched = zeros(N, 8, BckIdx+ForIdx);
    
    for i = 1:N
        startIdx = StimuliOnset(i)-BckIdx;
        endIdx = StimuliOnset(i)+ForIdx-1;
        
        epoched(i,:,:) = InputSignal(2:9, startIdx:endIdx);  
    end
end

function epoched = epochingTest(InputSignal,BackwardSamples,ForwardSamples,StimuliOnset)
    Fs = 3/(InputSignal(1,6)-InputSignal(1,3));
    
    BckIdx = floor(BackwardSamples*Fs);
    ForIdx = floor(ForwardSamples*Fs);
    N = length(StimuliOnset);
    
    epoched = zeros(N, 8 , BckIdx+ForIdx);
    
    for i = 1:N
        startIdx = StimuliOnset(i)-BckIdx;
        endIdx = StimuliOnset(i)+ForIdx-1;
        
        epoched(i,:,:) = InputSignal(2:9, startIdx:endIdx);  
    end
end


%%%%
function word = pred2word(index)

sc = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'];

word = [];

if(length(index) <= 8)
    word = sc(transpose(index));
     

else
    index1 = sort(index(1:2,1));
    index2 = sort(index(3:4,1));
    index3 = sort(index(5:6,1));
    index4 = sort(index(7:8,1));
    index5 = sort(index(9:10,1));
    
    index = [index1;index2;index3;index4;index5];
    
    c1 = ['A','G','M','S','Y','4'];
    c2 = ['B','H','N','T','Z','5'];
    c3 = ['C','I','O','U','0','6'];
    c4 = ['D','J','P','V','1','7'];
    c5 = ['E','K','Q','W','2','8'];
    c6 = ['F','L','R','X','3','9'];
    
    c = [c1;c2;c3;c4;c5];
    
    for k = 1:5
        word(k,1) = c(index(k,2)-6,index(k,1));
        
        
   
    end
    
    
end
end




