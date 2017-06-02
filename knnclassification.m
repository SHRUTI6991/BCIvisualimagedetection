clear;

% read in data
load trainingData_features;
load experimentData_features;

% init training data
classified = zeros(32,size(experimentData_features,1));
group = zeros(size(trainingData_features,1),1);

% v1
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8
training_data = zeros(size(trainingData_features,1),4);
experiment_data = zeros(size(experimentData_features,1),4);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(1,:) = label';

% v2
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8 purposely left the same as v1
training_data = zeros(size(trainingData_features,1),4);
experiment_data = zeros(size(experimentData_features,1),4);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(2,:) = label';

% v3
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8, 
% pband_O1, pband_O2, 
% pband_P7, pband_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,17); % pband_O1
   training_data(i,6) = trainingData_features(i,18); % pband_O2
   training_data(i,7) = trainingData_features(i,19); % pband_P7
   training_data(i,8) = trainingData_features(i,20); % pband_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8   
   experiment_data(i,5) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,6) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,7) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,8) =  experimentData_features(i,20); % pband_P8
   
   
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(3,:) = label';

% v4, 
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   training_data(i,13) = trainingData_features(i,9); % complexity_O1
   training_data(i,14) = trainingData_features(i,10); % complexity_O2
   training_data(i,15) = trainingData_features(i,11); % complexity_P7
   training_data(i,16) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,13) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(4,:) = label';

% v5, 
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8 purposely left the same as v4
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   training_data(i,13) = trainingData_features(i,9); % complexity_O1
   training_data(i,14) = trainingData_features(i,10); % complexity_O2
   training_data(i,15) = trainingData_features(i,11); % complexity_P7
   training_data(i,16) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,13) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(5,:) = label';

% v6, 
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),4);
experiment_data = zeros(size(experimentData_features,1),4);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,5); % mobility_O1
   training_data(i,2) = trainingData_features(i,6); % mobility_O2
   training_data(i,3) = trainingData_features(i,7); % mobility_P7
   training_data(i,4) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,2) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,3) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,4) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(6,:) = label';

% v7
% activity_O1, activity_O2
% activity_O1, activity_O2
training_data = zeros(size(trainingData_features,1),4);
experiment_data = zeros(size(experimentData_features,1),4);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);;
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(7,:) = label';

% v8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),4);
experiment_data = zeros(size(experimentData_features,1),4);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,9); % complexity_O1
   training_data(i,2) = trainingData_features(i,10); % complexity_O2
   training_data(i,3) = trainingData_features(i,11); % complexity_P7
   training_data(i,4) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,2) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,3) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,4) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(8,:) = label';

% v9
% activity_O1, activity_O2,
% activity_P7, activity_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,9); % complexity_O1
   training_data(i,6) = trainingData_features(i,10); % complexity_O2
   training_data(i,7) = trainingData_features(i,11); % complexity_P7
   training_data(i,8) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,6) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,7) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,8) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(9,:) = label';

% v10
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(10,:) = label';

% v11
% mobility_O1, mobility_O2, 
% mobility_P7, mobility_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,5); % mobility_O1
   training_data(i,2) = trainingData_features(i,6); % mobility_O2
   training_data(i,3) = trainingData_features(i,7); % mobility_P7
   training_data(i,4) = trainingData_features(i,8); % mobility_P8
   training_data(i,5) = trainingData_features(i,9); % complexity_O1
   training_data(i,6) = trainingData_features(i,10); % complexity_O2
   training_data(i,7) = trainingData_features(i,11); % complexity_P7
   training_data(i,8) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,2) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,3) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,4) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,5) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,6) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,7) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,8) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(11,:) = label';

% v12
% activity_O1, activity_O2,
% activity_P7, activity_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,1); % activity_O1
   training_data(i,2) = trainingData_features(i,2); % activity_O2
   training_data(i,3) = trainingData_features(i,3); % activity_P7
   training_data(i,4) = trainingData_features(i,4); % activity_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,2) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,3) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,4) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(12,:) = label';

% v13
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8,
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(13,:) = label';

% v14
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8,
% activity_O1, activity_O2 
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(14,:) = label';

% v15
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,9); % complexity_O1
   training_data(i,6) = trainingData_features(i,10); % complexity_O2
   training_data(i,7) = trainingData_features(i,11); % complexity_P7
   training_data(i,8) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,6) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,7) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,8) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(15,:) = label';

% v16
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% activity_O1, activity_O2, 
% activity_P7, activity_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(16,:) = label';

% v17
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% activity_O1, activity_O2, 
% activity_P7, activity_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(17,:) = label';

% v18
% p2p_O1, p2p_O2, 
% p2p_P7, p2p_P8, 
% mobility_O1, mobility_O2, 
% mobility_P7, mobility_P8, 
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,13); % p2p_O1
   training_data(i,2) = trainingData_features(i,14); % p2p_O2
   training_data(i,3) = trainingData_features(i,15); % p2p_P7
   training_data(i,4) = trainingData_features(i,16); % p2p_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,2) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,3) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,4) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(18,:) = label';

% v19
% pband_O1, pband_O2,
% pband_P7, pband_P8,
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(19,:) = label';

% v20
% pband_O1, pband_O2,
% pband_P7, pband_P8,
% activity_O1, activity_O2
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(20,:) = label';

% v21
% pband_O1, pband_O2
% pband_P7, pband_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),8);
experiment_data = zeros(size(experimentData_features,1),8);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,9); % complexity_O1
   training_data(i,6) = trainingData_features(i,10); % complexity_O2
   training_data(i,7) = trainingData_features(i,11); % complexity_P7
   training_data(i,8) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,6) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,7) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,8) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(21,:) = label';

% v22
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(22,:) = label';

% v23
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(23,:) = label';

% v24
% pband_O1, pband_O2
% pband_P7, pband_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8,
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,5); % mobility_O1
   training_data(i,6) = trainingData_features(i,6); % mobility_O2
   training_data(i,7) = trainingData_features(i,7); % mobility_P7
   training_data(i,8) = trainingData_features(i,8); % mobility_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,6) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,7) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,8) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(24,:) = label';

% v25
% pband_O1, pband_O2
% pband_P7, pband_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,1); % activity_O1
   training_data(i,6) = trainingData_features(i,2); % activity_O2
   training_data(i,7) = trainingData_features(i,3); % activity_P7
   training_data(i,8) = trainingData_features(i,4); % activity_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   training_data(i,13) = trainingData_features(i,9); % complexity_O1
   training_data(i,14) = trainingData_features(i,10); % complexity_O2
   training_data(i,15) = trainingData_features(i,11); % complexity_P7
   training_data(i,16) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,6) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,7) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,8) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,13) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(25,:) = label';

% v26
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8,
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(26,:) = label';

% v27
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,9); % complexity_O1
   training_data(i,10) = trainingData_features(i,10); % complexity_O2
   training_data(i,11) = trainingData_features(i,11); % complexity_P7
   training_data(i,12) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,10) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,11) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,12) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(27,:) = label';

% v28
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8
% activity_O1, activity_O2
% activity_P7, activity_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,1); % activity_O1
   training_data(i,10) = trainingData_features(i,2); % activity_O2
   training_data(i,11) = trainingData_features(i,3); % activity_P7
   training_data(i,12) = trainingData_features(i,4); % activity_P8
   training_data(i,13) = trainingData_features(i,9); % complexity_O1
   training_data(i,14) = trainingData_features(i,10); % complexity_O2
   training_data(i,15) = trainingData_features(i,11); % complexity_P7
   training_data(i,16) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,10) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,11) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,12) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,13) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(28,:) = label';

% v29
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8 
% activity_O1, activity_O2
% activity_P7, activity_P8
% mobility_O1, mobility_O2
% mobility_P7, mobility_P8
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,1); % activity_O1
   training_data(i,10) = trainingData_features(i,2); % activity_O2
   training_data(i,11) = trainingData_features(i,3); % activity_P7
   training_data(i,12) = trainingData_features(i,4); % activity_P8
   training_data(i,13) = trainingData_features(i,5); % mobility_O1
   training_data(i,14) = trainingData_features(i,6); % mobility_O2
   training_data(i,15) = trainingData_features(i,7); % mobility_P7
   training_data(i,16) = trainingData_features(i,8); % mobility_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,10) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,11) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,12) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,13) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,8); % mobility_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(29,:) = label';

% v30
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2,
% p2p_P7, p2p_P8
% mobility_O1, mobility_O2,
% mobility_P7, mobility_P8
% complexity_O1, complexity_O2
% complexity_P7, complexity_P8
training_data = zeros(size(trainingData_features,1),16);
experiment_data = zeros(size(experimentData_features,1),16);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,5); % mobility_O1
   training_data(i,10) = trainingData_features(i,6); % mobility_O2
   training_data(i,11) = trainingData_features(i,7); % mobility_P7
   training_data(i,12) = trainingData_features(i,8); % mobility_P8
   training_data(i,13) = trainingData_features(i,9); % complexity_O1
   training_data(i,14) = trainingData_features(i,10); % complexity_O2
   training_data(i,15) = trainingData_features(i,11); % complexity_P7
   training_data(i,16) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,10) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,11) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,12) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,13) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,14) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,15) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,16) =  experimentData_features(i,12); % complexity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(30,:) = label';

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
training_data = zeros(size(trainingData_features,1),20);
experiment_data = zeros(size(experimentData_features,1),20);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,1); % activity_O1
   training_data(i,10) = trainingData_features(i,2); % activity_O2
   training_data(i,11) = trainingData_features(i,3); % activity_P7
   training_data(i,12) = trainingData_features(i,4); % activity_P8
   training_data(i,13) = trainingData_features(i,5); % mobility_O1
   training_data(i,14) = trainingData_features(i,6); % mobility_O2
   training_data(i,15) = trainingData_features(i,7); % mobility_P7
   training_data(i,16) = trainingData_features(i,8); % mobility_P8
   training_data(i,17) = trainingData_features(i,9); % complexity_O1
   training_data(i,18) = trainingData_features(i,10); % complexity_O2
   training_data(i,19) = trainingData_features(i,11); % complexity_P7
   training_data(i,20) = trainingData_features(i,12); % complexity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,10) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,11) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,12) =  experimentData_features(i,4); % activity_P8
   experiment_data(i,13) =  experimentData_features(i,5); % mobility_O1
   experiment_data(i,14) =  experimentData_features(i,6); % mobility_O2 
   experiment_data(i,15) =  experimentData_features(i,7); % mobility_P7
   experiment_data(i,16) =  experimentData_features(i,8); % mobility_P8
   experiment_data(i,17) =  experimentData_features(i,9); %complexity_O1
   experiment_data(i,18) =  experimentData_features(i,10); % complexity_O2
   experiment_data(i,19) =  experimentData_features(i,11); % complexity_P7
   experiment_data(i,20) =  experimentData_features(i,12); % complexity_P8
   
   
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(31,:) = label';

% v32
% pband_O1, pband_O2
% pband_P7, pband_P8
% p2p_O1, p2p_O2
% p2p_P7, p2p_P8 
% activity_O1, activity_O2
% activity_P7, activity_P8
training_data = zeros(size(trainingData_features,1),12);
experiment_data = zeros(size(experimentData_features,1),12);
for i=1:size(trainingData_features,1)
   training_data(i,1) = trainingData_features(i,17); % pband_O1
   training_data(i,2) = trainingData_features(i,18); % pband_O2
   training_data(i,3) = trainingData_features(i,19); % pband_P7
   training_data(i,4) = trainingData_features(i,20); % pband_P8
   training_data(i,5) = trainingData_features(i,13); % p2p_O1
   training_data(i,6) = trainingData_features(i,14); % p2p_O2
   training_data(i,7) = trainingData_features(i,15); % p2p_P7
   training_data(i,8) = trainingData_features(i,16); % p2p_P8
   training_data(i,9) = trainingData_features(i,1); % activity_O1
   training_data(i,10) = trainingData_features(i,2); % activity_O2
   training_data(i,11) = trainingData_features(i,3); % activity_P7
   training_data(i,12) = trainingData_features(i,4); % activity_P8
   
   experiment_data(i,1) =  experimentData_features(i,17); % pband_O1
   experiment_data(i,2) =  experimentData_features(i,18); % pband_O2
   experiment_data(i,3) =  experimentData_features(i,19); % pband_P7
   experiment_data(i,4) =  experimentData_features(i,20); % pband_P8
   experiment_data(i,5) =  experimentData_features(i,13); % p2p_O1
   experiment_data(i,6) =  experimentData_features(i,14); % p2p_O2
   experiment_data(i,7) =  experimentData_features(i,15); % p2p_P7
   experiment_data(i,8) =  experimentData_features(i,16); % p2p_P8
   experiment_data(i,9) =  experimentData_features(i,1); % activity_O1
   experiment_data(i,10) =  experimentData_features(i,2); % activity_O2
   experiment_data(i,11) =  experimentData_features(i,3); % activity_P7
   experiment_data(i,12) =  experimentData_features(i,4); % activity_P8
   group(i) = trainingData_features(i,21);
end
% KNN Classification fitting
knn_model = ClassificationKNN.fit(training_data, group, 'NumNeighbors', 9);
% KNN Classification
[label,score,cost] = predict(knn_model,experiment_data);
classified(32,:) = label';


% save classification results
dlmwrite('KNN Classification Result.txt',classified(1,:),'newline','pc','delimiter','')
for i=2:32
    dlmwrite('KNN Classification Result.txt',classified(i,:),'-append','newline','pc','delimiter','')
end


