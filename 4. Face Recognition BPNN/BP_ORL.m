%% Project: Face Recognition using BPNN
%% Description: This project implements a backpropagation neural network for face classification using PCA-LDA features.
%% Reference: PCA-LDA feature extraction: https://www.mathworks.com/matlabcentral/answers/1677684-how-to-perform-template-matching-between-matrix-and-image

function BP_ORL

% input ORL training data after PCALDA, using function 'ORL_PCALDA.m'
[FACE, testFACE] = ORL_PCALDA;
input = [FACE;testFACE]; % training=input(1:100); testing=input(101:200)

% target
p = eye(20);
target = reshape(repmat(p,[5,1]),20,100)';
% initialize the numbers of neuron
inputneurons = 20;
a = inputneurons;
hiddenneurons = 40;
b = hiddenneurons;
outputneurons = 20; % How many people
c = outputneurons;
% learning rate
lr = 0.05; 

hiddenmatrix = zeros(a, b);
for i=1:1:a
    for j=1:1:b
        hiddenmatrix(i,j)=rand;
    end
end

outputmatrix = zeros(b,c);
for i=1:1:b
    for j=1:1:c
        outputmatrix(i,j)=rand;
    end
end

RMSE =zeros(100,1);

% Training
for epoch=1:1:200
    t=[];

    for iter=1:1:100 % Number of faces

        % 前傳部分
        hiddensigma = input(iter,:)*hiddenmatrix; % hiddensigma = 1x40
        hiddennet = logsig(hiddensigma);       
        outputsigma = hiddennet*outputmatrix; % outputsigma = 1x20
        outputnet = purelin(outputsigma);        
        
        % 倒傳部分
        % 輸出層的 delta
        doutputnet = dpurelin(outputsigma);
        deltaoutput = doutputnet.*(target(iter,:)-outputnet);
        error = target(iter,:) - outputnet;
        t = [t;sum(error.^2)];

        % 隱藏層的 delta
        temp = outputmatrix*deltaoutput';
        transfer = dlogsig(hiddensigma, hiddennet);
        deltahidden = temp.*transfer';
        
        % 輸出層權重更新
        newoutputmatrix = outputmatrix + lr*(deltaoutput'*hiddennet)';
        outputmatrix = newoutputmatrix;
        
        % 隱藏層權重更新 
        newhiddenmatrix = hiddenmatrix + lr*(deltahidden*input(iter,:))';
        hiddenmatrix=newhiddenmatrix;
    end
    RMSE(epoch) = sqrt(sum(t)/100);
    
    % Update learning rate: 
    % 如果RMSE比上次大,就把learning Rate減小10倍
    if epoch>=2 && RMSE(epoch)>=RMSE(epoch-1)
        lr = lr/10;
    end
    
    fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t)/100));
end

fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE(epoch));
plot(1:epoch,RMSE(1:epoch));
legend('Training');
ylabel('RMSE');xlabel('Epoch');

% Testing
Tot_Correct=0;

for i=1:1:100
    hiddensigma = input(i+100,:)*hiddenmatrix;
    hiddennet = logsig(hiddensigma);       
    outputsigma = hiddennet*outputmatrix;
    outputnet = purelin(outputsigma);
    [~,maxindex] = max(outputnet);
    if maxindex == ceil(i/5)
        Tot_Correct = Tot_Correct + 1;
    end
end

Tot_Correct
Tot_Percent= (Tot_Correct) / (length(input)-100);
Test_correct_percent = Tot_Percent



