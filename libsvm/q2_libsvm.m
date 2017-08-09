clear ; %clears all the variables 
close all; %closes extra windows
clc %clears the screen

%=============================================== (c) LIBSVM Linear & Gaussian Classification ============================================

x_train = load('../traindata.txt');		%loading training data
y_train = load('../trainlabels.txt');

x_test = load('../testdata.txt');
y_test = load('../testlabels.txt');

%--------------------------------------Linear Kernel-------------------------------------------------------

model_linear = svmtrain(y_train, x_train, '-t 0 -c 500');
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(y_test, x_test, model_linear);
%pause;
model_linear.sv_indices
%pause;

%--------------------------------------Gaussian Kernel-------------------------------------------------------

model_gaussian = svmtrain(y_train, x_train, '-t 2 -g 2.5 -c 500');
[predict_label_G, accuracy_G, dec_values_G] = svmpredict(y_test, x_test, model_gaussian);
%pause;
model_gaussian.sv_indices
%pause;

%========================================================= (d) Cross Validation ==========================================================

accuracy_cv_test = zeros(7,3);
accuracy_cv_test(:,1) = [1; 10; 100; 1000; 10000; 100000; 1000000];

for i = 1 : length(accuracy_cv_test)						
	accuracy_cv_test(i,2) = svmtrain(y_train, x_train, sprintf("-t %d -g %d -c %d -v %d", 2, 2.5, accuracy_cv_test(i,1), 10));
	m = svmtrain(y_train, x_train, sprintf("-t %d -g %d -c %d", 2, 2.5, accuracy_cv_test(i,1)));
	[predict_label_G, accuracy_G, dec_values_G] = svmpredict(y_test, x_test, m);
	accuracy_cv_test(i,3) = accuracy_G(1);
end

%---------------------------Plotting the average validation accuracy and test set accuracy with varying C------------------------------------

accuracy_cv_test
figure;

x = [0:1:6];

% Plot Examples
plot(x, accuracy_cv_test(:,2), 'r.','MarkerSize', 9);
xlabel('log(base 10) C');
hold on;
plot(x, accuracy_cv_test(:,3), 'b.','MarkerSize', 9);
hold off;

legend('Avg. CV Accuracy', 'Test Set Accuracy');
