clear;

% initialization
minPeakHeighThreshold = 200;
scale = 1;
waveletName = 'morl';

% open config.txt
fileID = fopen('config.txt');

% load EEG sampling duration from config.txt
for i=1:8
    configStatement = fgetl(fileID);
end
fclose(fileID);
k = strfind(configStatement, '=');

% convert sampling interval from string to integer
samplingDuration = str2num(configStatement((k+2):length(configStatement)))/1000;

% change to Training_Data folder
cd Training_Data


% get training file list
training_data_file_List = dir('*.csv');
num_of_training_data_files = length(training_data_file_List);
trainingData_features = zeros(num_of_training_data_files,21);

Fs = 128;             % Sampling Frequency
Fstop1 = 2;           % First Stopband Frequency
Fpass1 = 4;           % First Passband Frequency
Fpass2 = 8;           % Second Passband Frequency
Fstop2 = 10;          % Second Stopband Frequency
Astop1 = 60;          % First Stopband Attenuation (dB)
Apass  = 1;           % Passband Ripple (dB)
Astop2 = 80;          % Second Stopband Attenuation (dB)
match  = 'stopband';  % Band to match exactly

% Construct an FDESIGN object and call its CHEBY2 method.
h  = fdesign.bandpass(Fstop1, Fpass1, Fpass2, Fstop2, Astop1, Apass, ...
                      Astop2, Fs);
bpf = design(h, 'cheby2', 'MatchExactly', match);
% data= dir(familiar.csv);
% dataRowNumber = size(data,1);
% labels= rand(dataRowNumber,1) > 0.5;
% randomColumn = rand(dataRowNumber,1);
% 
% X = [ randomColumn data labels];
% 
% 
% SortedData = sort(X,1);
% 
% crossValidationFolds = 10;
% numberOfRowsPerFold = dataRowNumber / crossValidationFolds;
% 
% crossValidationTrainData = [];
% crossValidationTestData = [];
% for startOfRow = 1:numberOfRowsPerFold:dataRowNumber
%     testRows = startOfRow:startOfRow+numberOfRowsPerFold-1;
%     if (startOfRow == 1)
%         trainRows = [max(testRows)+1:dataRowNumber];
%         else
%         trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
%     end
%     crossValidationTrainData = [crossValidationTrainData ; SortedData(trainRows ,:)];
%     crossValidationTestData = [crossValidationTestData ;SortedData(testRows ,:)];
% 
% end

% Process training data signals for features
for i=1:num_of_training_data_files
   % output file to track progress for C# progress bar
   count = strcat('..\',num2str(i),'.count'); 
   save(count,'');
   
   % read in signals
   file_data = csvread(training_data_file_List(i).name,1,8);
   training_data_P7 =  file_data(:,1);
   training_data_O1 =  file_data(:,2);
   training_data_O2 =  file_data(:,3);
   training_data_P8 =  file_data(:,4);
   
   % baseline removal
   training_data_O1 = training_data_O1 - mean(training_data_O1);
   training_data_O2 = training_data_O2 - mean(training_data_O2);
   training_data_P7 = training_data_P7 - mean(training_data_P7);
   training_data_P8 = training_data_P8 - mean(training_data_P8);
   
   % Signal Preprocessing Phase
   pre_processing_O1 = filter(bpf,transpose(training_data_O1));
   pre_processing_O2 = filter(bpf,transpose(training_data_O2));
   pre_processing_P7 = filter(bpf,transpose(training_data_P7));
   pre_processing_P8 = filter(bpf,transpose(training_data_P8));
   
   % perform CWT
    pre_processing_O1 = cwt(pre_processing_O1,scale,waveletName);
    pre_processing_O2 = cwt(pre_processing_O2,scale,waveletName);
   pre_processing_P7 = cwt(pre_processing_P7,scale,waveletName);
   pre_processing_P8 = cwt(pre_processing_P8,scale,waveletName);
   
   % FFT Phase
   % Pxx = PSD of CWT coefficients, F = cyclical frequencies
   [Pxx_pre_processing_O1, F_pre_processing_O1] = periodogram(pre_processing_O1,rectwin(length(pre_processing_O1)),length(pre_processing_O1),Fs)
   pband_O1 = bandpower(Pxx_pre_processing_O1,F_pre_processing_O1,'psd')/10000;
   
   [Pxx_pre_processing_O2, F_pre_processing_O2] = periodogram(pre_processing_O2,rectwin(length(pre_processing_O2)),length(pre_processing_O2),Fs)
   pband_O2 = bandpower(Pxx_pre_processing_O2,F_pre_processing_O2,'psd')/10000;

   [Pxx_pre_processing_P7, F_pre_processing_P7] = periodogram(pre_processing_P7,rectwin(length(pre_processing_P7)),length(pre_processing_P7),Fs)
   pband_P7 = bandpower(Pxx_pre_processing_P7,F_pre_processing_P7,'psd')/10000;
   
   [Pxx_pre_processing_P8, F_pre_processing_P8] = periodogram(pre_processing_P8,rectwin(length(pre_processing_P8)),length(pre_processing_P8),Fs)
   pband_P8 = bandpower(Pxx_pre_processing_P8,F_pre_processing_P8,'psd')/10000;
   
   % computing Hjorth Parameters (Activity)   
   activity_O1 = var(pre_processing_O1)/10000000000;
   activity_O2 = var(pre_processing_O2)/10000000000;
   activity_P7 = var(pre_processing_P7)/10000000000;
   activity_P8 = var(pre_processing_P8)/10000000000;
   
   % computing Hjorth Parameters (Mobility)
   first_derivative_O1 = diff(pre_processing_O1)/(1/Fs);
   first_derivative_O2 = diff(pre_processing_O2)/(1/Fs);
   first_derivative_P7 = diff(pre_processing_P7)/(1/Fs);
   first_derivative_P8 = diff(pre_processing_P8)/(1/Fs);
   
   mobility_O1 = sqrt( var(first_derivative_O1)/var(pre_processing_O1) );
   mobility_O2 = sqrt( var(first_derivative_O2)/var(pre_processing_O2) );
   mobility_P7 = sqrt( var(first_derivative_P7)/var(pre_processing_P7) );
   mobility_P8 = sqrt( var(first_derivative_P8)/var(pre_processing_P8) );
   
   % computing Hjorth Parameters (Complexity)
   second_derivative_O1 = diff(first_derivative_O1)/(1/Fs);
   second_derivative_O2 = diff(first_derivative_O2)/(1/Fs);
   second_derivative_P7 = diff(first_derivative_P7)/(1/Fs);
   second_derivative_P8 = diff(first_derivative_P8)/(1/Fs);
   
   mobility_2_O1 = sqrt( var(second_derivative_O1)/var(first_derivative_O1) );
   mobility_2_O2 = sqrt( var(second_derivative_O2)/var(first_derivative_O2) );
   mobility_2_P7 = sqrt( var(second_derivative_P7)/var(first_derivative_P7) );
   mobility_2_P8 = sqrt( var(second_derivative_P8)/var(first_derivative_P8) );
   
   complexity_O1 = mobility_2_O1/mobility_O1;
   complexity_O2 = mobility_2_O2/mobility_O2;
   complexity_P7 = mobility_2_P7/mobility_P7;
   complexity_P8 = mobility_2_P8/mobility_P8;
   
   % Computing the peak-to-peak value
   p2p_O1 = peak2peak(pre_processing_O1);
   p2p_O2 = peak2peak(pre_processing_O2);
   p2p_P7 = peak2peak(pre_processing_P7);
   p2p_P8 = peak2peak(pre_processing_P8);
   
   % saving features
   fileName = training_data_file_List(i).name;
   trainingData_features(i,1) = activity_O1;
   trainingData_features(i,2) = activity_O2;
   trainingData_features(i,3) = activity_P7;
   trainingData_features(i,4) = activity_P8;
   a=[activity_O1;activity_O2;activity_P7;activity_P8]
   Data_features(i,1)= mean(a);
   trainingData_features(i,5) = mobility_O1;
   trainingData_features(i,6) = mobility_O2;
   trainingData_features(i,7) = mobility_P7;
   trainingData_features(i,8) = mobility_P8;
   b=[complexity_O1;complexity_O2;complexity_P7;complexity_P8]
    Data_features(i,2)=mean(b);
   trainingData_features(i,9) = complexity_O1;
   trainingData_features(i,10) = complexity_O2;
   trainingData_features(i,11) = complexity_P7;
   trainingData_features(i,12) = complexity_P8;
   c=[mobility_O1;mobility_O2;mobility_P7;mobility_P8]
   Data_features(i,3)= mean(c);
   trainingData_features(i,13) = p2p_O1;
   trainingData_features(i,14) = p2p_O2;
   trainingData_features(i,15) = p2p_P7;
   trainingData_features(i,16) = p2p_P8;
   d=[p2p_O1;p2p_O2;p2p_P7;p2p_P8]
    Data_features(i,4)= mean(d);
   trainingData_features(i,17) = pband_O1;
   trainingData_features(i,18) = pband_O2;
   trainingData_features(i,19) = pband_P7;
   trainingData_features(i,20) = pband_P8;
   e=[pband_O1;pband_O2;pband_P7;pband_P8]
    Data_features(i,5)= mean(e);
   
   
   if(strcmp('f',lower(fileName(1))))
       trainingData_features(i,21) = 1; % familiar stimulus
   else
       trainingData_features(i,21) = 0;
   end
end
% return back to original folder
cd ..

% export features into matfile
save('trainingData_features.mat','trainingData_features');