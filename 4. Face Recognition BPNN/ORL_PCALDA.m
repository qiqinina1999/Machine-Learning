function [prototypeFACE, testFACE] = ORL_PCALDA

people = 40;        % Total people for training
trainpeople = 20;   % How many people use to train and test
withinsample = 5;   % How many people will be used for training
FACEnumber = trainpeople*withinsample; % How many FACE will be use
principlenum = 50;  % Dimension after PCA transform
output_dim = 20;    % LDA output

FFACE=[];

for k=1:1: people
     for m=1:2:10
         matchstring=['..\orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
         matchX=imread(matchstring);
         matchX=double(matchX);
         if (k==1 && m==1)
            [row,col]=size(matchX);
         end         
         matchtempF=[];
         %--arrange the image into a vector
         for n=1:row
             matchtempF=[matchtempF, matchX(n,:)];
         end
         
         FFACE = [FFACE; matchtempF];  % FFACE = 200x1024
     end
end

TotalMeanFACE=mean(FFACE);
zeromeanTotalFACE=FFACE;

% zero mean
for i=1:1:withinsample*people
    for j=1:1:(row)*(col) 
        zeromeanTotalFACE(i,j)=zeromeanTotalFACE(i,j)-TotalMeanFACE(j); %¥¿³W¤Æ
    end
end

SST = zeromeanTotalFACE' * zeromeanTotalFACE;

[PCA, latent] = eig(SST);
eigvalue = diag(latent);
[~, index] = sort(eigvalue, 'descend');
PCA = PCA(:, index);
eigvalue = eigvalue(index);
projectPCA = PCA(:, 1:principlenum);

pcaTotalFACE = [];
for i = 1:1:withinsample*people
    tempFACE = zeromeanTotalFACE(i,:);
    tempFACE = tempFACE*projectPCA;
    pcaTotalFACE = [pcaTotalFACE; tempFACE]; % pcaTotalFACE will be 200x50
end

% --------------------LDA Transform--------------------
% ------------(Linear Discriminant Analysis)-----------
for i = 1:withinsample:withinsample*people %%%
    withinFACE = pcaTotalFACE(i:i+withinsample-1,:);
    if (i==1)
        SW = withinFACE'*withinFACE;  %SW=cov(withinFACE)
        ClassMean = mean(withinFACE);
    end
    if (i>1)
        SW = SW + (withinFACE'*withinFACE); %SW=SW+cov(withinFACE)
        ClassMean = [ClassMean; mean(withinFACE)];
    end
end

pcaTotalmean = mean(pcaTotalFACE); % pcaTotalmean = 1x50

SB = ClassMean'*ClassMean; %SB=cov(ClassMean)

[eigvector, eigvalue] = eig(inv(SW)*SB);  % arg max(SB/SW) --Fisher' Criterion
eigvalue = diag(eigvalue);
[~, index] = sort(eigvalue, 'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

prototypeFACE = pcaTotalFACE*eigvector(:,1:output_dim);
prototypeFACE = prototypeFACE(1:FACEnumber,:);

% Test data
testFACE = [];
TFACE = [];
for k=1:1: people
     for m=2:2:10
         matchstring=['..\orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
         test=imread(matchstring);
         test=double(test);
         tempF = [];
         for p = 1:row
            tempF = [tempF, test(p,:)]; % Turn into 1x1024
         end
         TFACE = [TFACE; tempF];
         % PCA Transformation
         tempF = tempF-TotalMeanFACE;
         tempF = tempF*projectPCA; %row
         % LDA Transformation
         resultF = tempF*eigvector(:,1:output_dim);
         % Collect all face
         testFACE = [testFACE; resultF];
     end
end
testFACE = testFACE(1:FACEnumber,:);
end




