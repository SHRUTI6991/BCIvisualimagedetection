clear;

% read in data
load trainingData_features;
load experimentData_features;

cd Training_Data

% init training data
classified = zeros(32,size(trainingData_features,1));
group = zeros(size(trainingData_features,1),1);

% v1
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8


training_data = zeros(size(trainingData_features,1),6);
experiment_data = zeros(size(experimentData_features,1),6);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
     training_data(i,6) = trainingData_features(i,24); % p2p_T8
   
   experiment_data(i,1) =  experimentData_features(i,19); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,20); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,21); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,22); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(1,:) = svmclassify(SVMModel, experiment_data).';

% v2
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8 purposely left the same as v1
training_data = zeros(size(trainingData_features,1),6);
experiment_data = zeros(size(experimentData_features,1),6);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % p2p_P8
    training_data(i,5) = trainingData_features(i,29); % p2p_T7
     training_data(i,6) = trainingData_features(i,30); % p2p_T8
   
   experiment_data(i,1) =  experimentData_features(i,25); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,26); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,27); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,28); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,29); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,30); % p2p_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(2,:) = svmclassify(SVMModel, experiment_data).';

% v3
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8, 
% pband_O1, pband_O2, 
% pband_P7, pband_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
   training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,25); % pband_O1
   training_data(i,8) = trainingData_features(i,26); % pband_O2
   training_data(i,9) = trainingData_features(i,27); % pband_P7
   training_data(i,10) = trainingData_features(i,28); % pband_P8
    training_data(i,11) = trainingData_features(i,29); % pband_t7
   training_data(i,12) = trainingData_features(i,30); % pband_t8
   
   experiment_data(i,1) =  experimentData_features(i,19); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,20); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,21); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,22); % p2p_P8 
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   
   experiment_data(i,7) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,8) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,9) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,10) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,11) =  experimentData_features(i,29); % pband_t7
   experiment_data(i,12) =  experimentData_features(i,30); % pband_t8
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(3,:) = svmclassify(SVMModel, experiment_data).';

% v4, 
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
   training_data(i,19) = trainingData_features(i,13); % complexity_O1
   training_data(i,20) = trainingData_features(i,14); % complexity_O2
   training_data(i,21) = trainingData_features(i,15); % complexity_P7
   training_data(i,22) = trainingData_features(i,16); % complexity_P8
    training_data(i,23) = trainingData_features(i,17); % complexity_T7
   training_data(i,24) = trainingData_features(i,18); % complexity_T8
   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,19) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,20) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,21) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,22) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,23) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,24) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(4,:) = svmclassify(SVMModel, experiment_data).';

% v5, 
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8 purposely left the same as v4
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
   training_data(i,19) = trainingData_features(i,13); % complexity_O1
   training_data(i,20) = trainingData_features(i,14); % complexity_O2
   training_data(i,21) = trainingData_features(i,15); % complexity_P7
   training_data(i,22) = trainingData_features(i,16); % complexity_P8
    training_data(i,23) = trainingData_features(i,17); % complexity_T7
   training_data(i,24) = trainingData_features(i,18); % complexity_T8
   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,19) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,20) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,21) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,22) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,23) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,24) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(5,:) = svmclassify(SVMModel, experiment_data).';

% v6, 
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),6);
experiment_data = zeros(size(experimentData_features,1),6);
for i=1:size(trainingData_features,1)
  
    training_data(i,1) = trainingData_features(i,7); % mobility_O1
   training_data(i,2) = trainingData_features(i,8); % mobility_O2
   training_data(i,3) = trainingData_features(i,9); % mobility_P7
   training_data(i,4) = trainingData_features(i,10); % mobility_P8
     training_data(i,5) = trainingData_features(i,11); % mobility_T7
   training_data(i,6) = trainingData_features(i,12); % mobility_T8
    
   experiment_data(i,1) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,2) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,3) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,4) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,5) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,6) =  experimentData_features(i,12); % mobility_T8
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(6,:) = svmclassify(SVMModel, experiment_data).';

% v7
% activity_O1, activity_O2
% activity_O1, activity_O2
training_data = zeros(size(trainingData_features,1),6);
experiment_data = zeros(size(experimentData_features,1),6);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % activity_T7
   training_data(i,6) = trainingData_features(i,6); % activity_T8
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,6) =  experimentData_features(i,6); % activity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(7,:) = svmclassify(SVMModel, experiment_data).';

% v8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),6);
experiment_data = zeros(size(experimentData_features,1),6);
for i=1:size(trainingData_features,1)
 training_data(i,1) = trainingData_features(i,13); % complexity_O1
   training_data(i,2) = trainingData_features(i,14); % complexity_O2
   training_data(i,3) = trainingData_features(i,15); % complexity_P7
   training_data(i,4) = trainingData_features(i,16); % complexity_P8
    training_data(i,5) = trainingData_features(i,17); % complexity_T7
   training_data(i,6) = trainingData_features(i,18); % complexity_T8
   
   experiment_data(i,1) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,2) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,3) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,4) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,5) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,6) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(8,:) = svmclassify(SVMModel, experiment_data).';

% v9
% activity_O1, activity_O2,
% activity_P7, activity_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
      training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % activity_T7
   training_data(i,6) = trainingData_features(i,6); % activity_T8
   training_data(i,7) = trainingData_features(i,13); % complexity_O1
   training_data(i,8) = trainingData_features(i,14); % complexity_O2
   training_data(i,9) = trainingData_features(i,15); % complexity_P7
   training_data(i,10) = trainingData_features(i,16); % complexity_P8
    training_data(i,11) = trainingData_features(i,17); % complexity_T7
   training_data(i,12) = trainingData_features(i,18); % complexity_T8
   
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
     experiment_data(i,5) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,6) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,7) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,8) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,9) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,10) =  experimentData_features(i,16); % complexity_P8
   experiment_data(i,11) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,12) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(9,:) = svmclassify(SVMModel, experiment_data).';

% v10
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % activity_T7
   training_data(i,6) = trainingData_features(i,6); % activity_T8
   training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
     training_data(i,11) = trainingData_features(i,11); % mobility_T7
   training_data(i,12) = trainingData_features(i,12); % mobility_T8
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,6) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(10,:) = svmclassify(SVMModel, experiment_data).';

% v11
% mobility_O1, mobility_O2, 
% mobility_P7, mobility_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
  training_data(i,1) = trainingData_features(i,7); % mobility_O1
   training_data(i,2) = trainingData_features(i,8); % mobility_O2
   training_data(i,3) = trainingData_features(i,9); % mobility_P7
   training_data(i,4) = trainingData_features(i,10); % mobility_P8
     training_data(i,5) = trainingData_features(i,11); % mobility_T7
   training_data(i,6) = trainingData_features(i,12); % mobility_T8
   training_data(i,7) = trainingData_features(i,13); % complexity_O1
   training_data(i,8) = trainingData_features(i,14); % complexity_O2
   training_data(i,9) = trainingData_features(i,15); % complexity_P7
   training_data(i,10) = trainingData_features(i,16); % complexity_P8
    training_data(i,11) = trainingData_features(i,17); % complexity_T7
   training_data(i,12) = trainingData_features(i,18); % complexity_T8
  
   experiment_data(i,1) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,2) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,3) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,4) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,5) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,6) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,7) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,8) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,9) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,10) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,11) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,12) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(11,:) = svmclassify(SVMModel, experiment_data).';

% v12
% activity_O1, activity_O2,
% activity_P7, activity_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
      training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % activity_T7
   training_data(i,6) = trainingData_features(i,6); % activity_T8
   training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
     training_data(i,11) = trainingData_features(i,11); % mobility_T7
   training_data(i,12) = trainingData_features(i,12); % mobility_T8
   training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8
  
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,6) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(12,:) = svmclassify(SVMModel, experiment_data).';

% v13
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8,
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   
   training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
     training_data(i,11) = trainingData_features(i,11); % mobility_T7
   training_data(i,12) = trainingData_features(i,12); % mobility_T8
  
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(13,:) = svmclassify(SVMModel, experiment_data).';

% v14
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8,
% activity_O1, activity_O2 
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
    training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
  
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
  
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(14,:) = svmclassify(SVMModel, experiment_data).';

% v15
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
  
   training_data(i,7) = trainingData_features(i,13); % complexity_O1
   training_data(i,8) = trainingData_features(i,14); % complexity_O2
   training_data(i,9) = trainingData_features(i,15); % complexity_P7
   training_data(i,10) = trainingData_features(i,16); % complexity_P8
    training_data(i,11) = trainingData_features(i,17); % complexity_T7
   training_data(i,12) = trainingData_features(i,18); % complexity_T8
   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   
   experiment_data(i,7) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,8) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,9) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,10) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,11) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,12) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(15,:) = svmclassify(SVMModel, experiment_data).';

% v16
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
  
   training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8
   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
  
   experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(16,:) = svmclassify(SVMModel, experiment_data).';

% v17
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% activity_O1, activity_O2, 
% activity_P7, activity_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8

   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
  
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(17,:) = svmclassify(SVMModel, experiment_data).';

% v18
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% mobility_O1, mobility_O2, 
% mobility_P7, mobility_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,19); % p2p_O1
   training_data(i,2) = trainingData_features(i,20); % p2p_O2
   training_data(i,3) = trainingData_features(i,21); % p2p_P7
   training_data(i,4) = trainingData_features(i,22); % p2p_P8
    training_data(i,5) = trainingData_features(i,23); % p2p_T7
   training_data(i,6) = trainingData_features(i,24); % p2p_T8
  
   training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
     training_data(i,11) = trainingData_features(i,11); % mobility_T7
   training_data(i,12) = trainingData_features(i,12); % mobility_T8
   training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8
   
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,6) =  experimentData_features(i,24); % p2p_T8
  
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(18,:) = svmclassify(SVMModel, experiment_data).';

% v19
% pband_O1, pband_O2,
% pband_P7, pband_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)

   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
    training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
   training_data(i,11) = trainingData_features(i,11); % mobility_P7
   training_data(i,12) = trainingData_features(i,12); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(19,:) = svmclassify(SVMModel, experiment_data).';

% v20
% pband_O1, pband_O2,
% pband_P7, pband_P8,
% activity_O1, activity_O2
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   
   
    experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(20,:) = svmclassify(SVMModel, experiment_data).';

% v21
% pband_O1, pband_O2
% pband_P7, pband_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
     training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,13); % complexity_O1
   training_data(i,8) = trainingData_features(i,14); % complexity_O2
   training_data(i,9) = trainingData_features(i,15); % complexity_P7
   training_data(i,10) = trainingData_features(i,16); % complexity_P8
    training_data(i,11) = trainingData_features(i,17); % complexity_T7
   training_data(i,12) = trainingData_features(i,18); % complexity_T8
   
  
   
    experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
  experiment_data(i,7) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,8) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,9) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,10) =  experimentData_features(i,16); % complexity_P8
   experiment_data(i,11) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,12) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(21,:) = svmclassify(SVMModel, experiment_data).';

% v22
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
  
   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   
    training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8
  
  
  
   experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
  experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(22,:) = svmclassify(SVMModel, experiment_data).';

% v23
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
  raining_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
 
 experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
     experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(23,:) = svmclassify(SVMModel, experiment_data).';

% v24
% pband_O1, pband_O2
% pband_P7, pband_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
    training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
    training_data(i,7) = trainingData_features(i,7); % mobility_O1
   training_data(i,8) = trainingData_features(i,8); % mobility_O2
   training_data(i,9) = trainingData_features(i,9); % mobility_P7
   training_data(i,10) = trainingData_features(i,10); % mobility_P8
   training_data(i,11) = trainingData_features(i,11); % mobility_P7
   training_data(i,12) = trainingData_features(i,12); % mobility_P8
   training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8

   
   experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,9) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,10) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,11) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,12) =  experimentData_features(i,12); % mobility_T8
      
   experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(24,:) = svmclassify(SVMModel, experiment_data).';

% v25
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
 training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,1); % activity_O1
   training_data(i,8) = trainingData_features(i,2); % activity_O2
   training_data(i,9) = trainingData_features(i,3); % activity_P7
   training_data(i,10) = trainingData_features(i,4); % activity_P8
   training_data(i,11) = trainingData_features(i,5); % activity_T7
   training_data(i,12) = trainingData_features(i,6); % activity_T8
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
   training_data(i,19) = trainingData_features(i,13); % complexity_O1
   training_data(i,20) = trainingData_features(i,14); % complexity_O2
   training_data(i,21) = trainingData_features(i,15); % complexity_P7
   training_data(i,22) = trainingData_features(i,16); % complexity_P8
    training_data(i,23) = trainingData_features(i,17); % complexity_T7
   training_data(i,24) = trainingData_features(i,18); % complexity_T8
      experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
   
   experiment_data(i,7) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,8) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,9) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,10) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,11) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,12) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
   experiment_data(i,19) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,20) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,21) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,22) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,23) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,24) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(25,:) = svmclassify(SVMModel, experiment_data).';

% v26
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8,
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
  
  
      
   experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
   
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
   
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(26,:) = svmclassify(SVMModel, experiment_data).';

% v27
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
  training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,13); % complexity_O1
   training_data(i,14) = trainingData_features(i,14); % complexity_O2
   training_data(i,15) = trainingData_features(i,15); % complexity_P7
   training_data(i,16) = trainingData_features(i,16); % complexity_P8
    training_data(i,17) = trainingData_features(i,17); % complexity_T7
   training_data(i,18) = trainingData_features(i,18); % complexity_T8
  
 experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
      
   experiment_data(i,13) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,17) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,18) =  experimentData_features(i,18); % complexity_T8
   
   
   
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(27,:) = svmclassify(SVMModel, experiment_data).';

% v28
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
 training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,1); % activity_O1
   training_data(i,14) = trainingData_features(i,2); % activity_O2
   training_data(i,15) = trainingData_features(i,3); % activity_P7
   training_data(i,16) = trainingData_features(i,4); % activity_P8
   training_data(i,17) = trainingData_features(i,5); % activity_T7
   training_data(i,18) = trainingData_features(i,6); % activity_T8
   training_data(i,19) = trainingData_features(i,13); % complexity_O1
   training_data(i,20) = trainingData_features(i,14); % complexity_O2
   training_data(i,21) = trainingData_features(i,15); % complexity_P7
   training_data(i,22) = trainingData_features(i,16); % complexity_P8
    training_data(i,23) = trainingData_features(i,17); % complexity_T7
   training_data(i,24) = trainingData_features(i,18); % complexity_T8
   
  
   
 
    experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
 experiment_data(i,13) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,14) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,15) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,16) =  experimentData_features(i,4); % activity_P8
     experiment_data(i,17) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,18) =  experimentData_features(i,6); % activity_T8
   experiment_data(i,19) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,20) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,21) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,22) =  experimentData_features(i,16); % complexity_P8
   experiment_data(i,23) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,24) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(28,:) = svmclassify(SVMModel, experiment_data).';

% v29
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8 
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,1); % activity_O1
   training_data(i,14) = trainingData_features(i,2); % activity_O2
   training_data(i,15) = trainingData_features(i,3); % activity_P7
   training_data(i,16) = trainingData_features(i,4); % activity_P8
   training_data(i,17) = trainingData_features(i,5); % activity_T7
   training_data(i,18) = trainingData_features(i,6); % activity_T8
    training_data(i,19) = trainingData_features(i,7); % mobility_O1
   training_data(i,20) = trainingData_features(i,8); % mobility_O2
   training_data(i,21) = trainingData_features(i,9); % mobility_P7
   training_data(i,22) = trainingData_features(i,10); % mobility_P8
     training_data(i,23) = trainingData_features(i,11); % mobility_T7
   training_data(i,24) = trainingData_features(i,12); % mobility_T8
    
experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
 experiment_data(i,13) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,14) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,15) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,16) =  experimentData_features(i,4); % activity_P8
     experiment_data(i,17) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,18) =  experimentData_features(i,6); % activity_T8
   
   experiment_data(i,19) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,20) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,21) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,22) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,23) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,24) =  experimentData_features(i,12); % mobility_T8
  
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(29,:) = svmclassify(SVMModel, experiment_data).';

% v30
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),24);
experiment_data = zeros(size(experimentData_features,1),24);
for i=1:size(trainingData_features,1)
  training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,7); % mobility_O1
   training_data(i,14) = trainingData_features(i,8); % mobility_O2
   training_data(i,15) = trainingData_features(i,9); % mobility_P7
   training_data(i,16) = trainingData_features(i,10); % mobility_P8
     training_data(i,17) = trainingData_features(i,11); % mobility_T7
   training_data(i,18) = trainingData_features(i,12); % mobility_T8
 
 training_data(i,19) = trainingData_features(i,13); % complexity_O1
   training_data(i,20) = trainingData_features(i,14); % complexity_O2
   training_data(i,21) = trainingData_features(i,15); % complexity_P7
   training_data(i,22) = trainingData_features(i,16); % complexity_P8
    training_data(i,23) = trainingData_features(i,17); % complexity_T7
   training_data(i,24) = trainingData_features(i,18); % complexity_T8
   
  
   
 
    experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
   
   experiment_data(i,13) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,18) =  experimentData_features(i,12); % mobility_T8
 
   experiment_data(i,19) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,20) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,21) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,22) =  experimentData_features(i,16); % complexity_P8
   experiment_data(i,23) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,24) =  experimentData_features(i,18); % complexity_T8
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(30,:) = svmclassify(SVMModel, experiment_data).';

% v31, 
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),30);
experiment_data = zeros(size(experimentData_features,1),30);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,1); % activity_O1
   training_data(i,14) = trainingData_features(i,2); % activity_O2
   training_data(i,15) = trainingData_features(i,3); % activity_P7
   training_data(i,16) = trainingData_features(i,4); % activity_P8
   training_data(i,17) = trainingData_features(i,5); % activity_T7
   training_data(i,18) = trainingData_features(i,6); % activity_T8
    training_data(i,19) = trainingData_features(i,7); % mobility_O1
   training_data(i,20) = trainingData_features(i,8); % mobility_O2
   training_data(i,21) = trainingData_features(i,9); % mobility_P7
   training_data(i,22) = trainingData_features(i,10); % mobility_P8
     training_data(i,23) = trainingData_features(i,11); % mobility_T7
   training_data(i,24) = trainingData_features(i,12); % mobility_T8
   training_data(i,25) = trainingData_features(i,13); % complexity_O1
   training_data(i,26) = trainingData_features(i,14); % complexity_O2
   training_data(i,27) = trainingData_features(i,15); % complexity_P7
   training_data(i,28) = trainingData_features(i,16); % complexity_P8
    training_data(i,29) = trainingData_features(i,17); % complexity_T7
   training_data(i,30) = trainingData_features(i,18); % complexity_T8

    
experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
 experiment_data(i,13) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,14) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,15) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,16) =  experimentData_features(i,4); % activity_P8
     experiment_data(i,17) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,18) =  experimentData_features(i,6); % activity_T8
   
   experiment_data(i,19) =  experimentData_features(i,7); % mobility_O1
   experiment_data(i,20) =  experimentData_features(i,8); % mobility_O2 
   experiment_data(i,21) =  experimentData_features(i,9); % mobility_P7
   experiment_data(i,22) =  experimentData_features(i,10); % mobility_P8
   experiment_data(i,23) =  experimentData_features(i,11); % mobility_T7
   experiment_data(i,24) =  experimentData_features(i,12); % mobility_T8
      
   experiment_data(i,25) =  experimentData_features(i,13); %complexity_O1
   experiment_data(i,26) =  experimentData_features(i,14); % complexity_O2
   experiment_data(i,27) =  experimentData_features(i,15); % complexity_P7
   experiment_data(i,28) =  experimentData_features(i,16); % complexity_P8
    experiment_data(i,29) =  experimentData_features(i,17); % complexity_T7
   experiment_data(i,30) =  experimentData_features(i,18); % complexity_T8
  
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(31,:) = svmclassify(SVMModel, experiment_data).';

% v32
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8 
% activity_O1, activity_O2
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),18);
experiment_data = zeros(size(experimentData_features,1),18);
for i=1:size(trainingData_features,1)
  training_data(i,1) = trainingData_features(i,25); % pband_O1
   training_data(i,2) = trainingData_features(i,26); % pband_O2
   training_data(i,3) = trainingData_features(i,27); % pband_P7
   training_data(i,4) = trainingData_features(i,28); % pband_P8
    training_data(i,5) = trainingData_features(i,29); % pband_T7
   training_data(i,6) = trainingData_features(i,30); % pband_T8
   training_data(i,7) = trainingData_features(i,19); % p2p_O1
   training_data(i,8) = trainingData_features(i,20); % p2p_O2
   training_data(i,9) = trainingData_features(i,21); % p2p_P7
   training_data(i,10) = trainingData_features(i,22); % p2p_P8
    training_data(i,11) = trainingData_features(i,23); % p2p_T7
   training_data(i,12) = trainingData_features(i,24); % p2p_T8
   training_data(i,13) = trainingData_features(i,1); % activity_O1
   training_data(i,14) = trainingData_features(i,2); % activity_O2
   training_data(i,15) = trainingData_features(i,3); % activity_P7
   training_data(i,16) = trainingData_features(i,4); % activity_P8
   training_data(i,17) = trainingData_features(i,5); % activity_T7
   training_data(i,18) = trainingData_features(i,6); % activity_T8
   
    
experiment_data(i,1) =  experimentData_features(i,25); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,26); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,27); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,28); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,29); % pband_T7
   experiment_data(i,6) =  experimentData_features(i,30); % pband_T8
    
   experiment_data(i,7) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,8) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,9) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,10) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,11) =  experimentData_features(i,23); % p2p_T7
   experiment_data(i,12) =  experimentData_features(i,24); % p2p_T8
 experiment_data(i,13) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,14) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,15) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,16) =  experimentData_features(i,4); % activity_P8
     experiment_data(i,17) =  experimentData_features(i,5); % activity_T7
   experiment_data(i,18) =  experimentData_features(i,6); % activity_T8
   
   group(i) = trainingData_features(i,31);
end
% SVM modelling
SVMModel = svmtrain(training_data, group);
classified(32,:) = svmclassify(SVMModel, experiment_data).';


% save classification results
dlmwrite('SVM Classification Result1.txt',classified(1,:),'newline','pc','delimiter','')
for i=2:32
    dlmwrite('SVM Classification Result1.txt',classified(i,:),'-append','newline','pc','delimiter','')
end


