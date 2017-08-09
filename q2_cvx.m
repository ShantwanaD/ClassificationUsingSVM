clear ; %clears all the variables 
close all; %closes extra windows
clc %clears the screen
% ======================================= Part 2: FACIAL ATTRACTIVENESS CLASSIFICATION  ============================================

% --------------------------------------------------(a) Support Vectors--------------------------------------------------

x_train = load('traindata.txt');		%loading training data
y_train = load('trainlabels.txt');
y_train(find(y_train == 2)) = -1;

m_train = length(y_train);			%no. of training examples
C = 500;

tmp = diag(y_train) * x_train;
Q = -(tmp * tmp') / 2;			%to express SVM dual problem in the form that cvx can take
b = ones(m_train,1);
c = 0; 

cvx_begin
    variable a(m_train)
    maximize((a' * Q * a) + (b' * a) + c)		%SVM dual problem
    subject to
        0 <= a <= C							%conditions
        a' * y_train == 0
cvx_end

sv_l = find(a > 1)				%support vector indices
nsv_l = numel(sv_l)						%no. of support vectors

% ---------------------------------------(b) Weight, Intercept & Test Accuracy --------------------------------------------

w = x_train' * (a .* y_train)			%weight vector(n X 1)

tmp = x_train * w;

m = find(a > 1 & a < 499);				%finding indices of the examples that lie on the margin
pos = m(find(y_train(m) == 1));				%dividing m according to their labels
neg = m(find(y_train(m) == -1));

b = -(max(tmp(neg,:)) + min(tmp(pos,:))) / 2		%intercept term

x_test = load('testdata.txt');
y_test = load('testlabels.txt');
y_test(find(y_test == 2)) = -1;

m_test = length(y_test);			%no. of test examples

temp = (x_test * w) + b;
count = 0;						%counter to count the no. of test examples that are predicted correctly
for i = 1 : m_test
	if (((temp(i) >= 0) & (y_test(i) == 1)) | ((temp(i) < 0) & (y_test(i) == -1)))
		count = count + 1;
	end
end
accuracy_linear_cvx = count * 100 / m_test			%test set accuracy for linear SVM

% -------------------------------------------------(c) Gaussian Kernel --------------------------------------------------

gamma = 2.5;
b = ones(m_train,1);
kmat = zeros(m_train,m_train);

for i = 1 : m_train			%filling the elements of matrix Q
	for j = 1 : m_train
	d = (x_train(i,:) - x_train(j,:))' ;
	kmat(i,j) = exp(-gamma * d' * d) ;
	Q(i,j) = -(kmat(i,j) * y_train(i) * y_train(j)) / 2 ;	
	end
end

cvx_begin
    variable a(m_train)
    maximize((a' * Q * a) + (b' * a) + c)		%SVM dual problem
    subject to
        0 <= a <= C							%conditions
        a' * y_train == 0
cvx_end

sv_g = find(a > 1)				%support vector indices
nsv_g = numel(sv_g)						%no. of support vectors

wtx = kmat' * (a .* y_train);

m = find(a > 1 & a < 499);				%finding indices of the examples that lie on the margin
pos = m(find(y_train(m) == 1));				%dividing m according to their labels
neg = m(find(y_train(m) == -1));

b = -(max(wtx(neg,:)) + min(wtx(pos,:))) / 2;		%intercept term

kmat = zeros(m_train,m_test);			
wtx_test = zeros(m_test,1);
for i = 1 : m_train			%filling the elements of kmat
	for j = 1 : m_test
	d = (x_train(i,:) - x_test(j,:))' ;
	kmat(i,j) = exp(-gamma * d' * d) ;
	wtx_test(j) = wtx_test(j) + (a(i) * y_train(i) * kmat(i,j));
	end
end

temp = wtx_test + b;
count = 0;						%counter to count the no. of test examples that are predicted correctly
for i = 1 : m_test
	if (((temp(i) >= 0) & (y_test(i) == 1)) | ((temp(i) < 0) & (y_test(i) == -1)))
		count = count + 1;
	end
end
accuracy_gaussian_cvx = count * 100 / m_test			%test set accuracy using gaussian kernel
