POINTS = 1000; % number of points to plot

% define the two random variables
% -------------------------------
for i=1:POINTS
	A(i) = round(rand*99)-50;              % A
	B(i) = round(rand*99)-50;              % B
end

%% plot the time course of the signals
% ------------------------------------

figure; subplot(2,1,1)
plot(A,'b')
title('Source signal 1 (s_{1})');
subplot(2,1,2)
plot(B,'r')
title('Source signal 2 (s_{2})');

%% Plot the source space and mixture space 

figure; subplot(1,2,1)
plot(A,B, '.');                        % plot the variables
set(gca, 'xlim', [-80 80], 'ylim', [-80 80]);  % redefines limits of the graph

% mix linearly these two variables
% --------------------------------
mix1 = [1.54 2.84; 1.42 0.3];
M1 = mix1(1,1)*A + mix1(1,2)*B;                          % mixing 1
M2 = mix1(2,1)*A + mix1(2,2)*B; % mixing 2

s1ax = [100;0];
s2ax = [0;100];
s1trans = mix1*s1ax;
s2trans = mix1*s2ax;

subplot(1,2,2); plot(M1,M2, '.');                      % plot the mixing
hold on
plot(s1trans(1),s1trans(2),'r*')
hold on
plot(s2trans(1), s2trans(2),'k*')
set(gca, 'ylim', get(gca, 'xlim'));            % redefines limits of the graph

%% plot the time course of the signal mixtures
figure; subplot(2,1,1)
plot(M1,'b')
title('Signal mixture 1 (x_{1})');
subplot(2,1,2)
plot(M2,'r'); 
title('Signal mixture 2 (x_{2})');


%% whiten the data
% ---------------
x = [M1;M2];
figure; subplot(1,3,1)
plot(x(1,:),x(2,:),'.k');
title('Mixture Signals');
c=cov(x');		 % covariance
sq=inv(sqrtm(c));        % inverse of square root
mx=mean(x');             % mean
xx=x-mx'*ones(1,POINTS); % subtract the mean

subplot(1,3,2)
plot(xx(1,:),xx(2,:),'.r')
title('Step 1: Centering Data');

xx=2*sq*xx;              
cov(xx')              % the covariance is now a diagonal matrix

subplot(1,3,3); plot(xx(2,:), xx(1,:), '.b');
title('Step 2: Whiten Data');

%% show projections
% ----------------
figure; 
axes('position', [0.2 0.2 0.8 0.8]); plot(xx(1,:), xx(2,:), '.'); hold on;
axes('position', [0   0.2 0.2 0.8]); hist(xx(1,:));  set(gca, 'view', [90 90]);
axes('position', [0.2 0   0.8 0.2]); hist(xx(2,:));

% show projections
% ----------------
figure; 
axes('position', [0.2 0.2 0.8 0.8]); plot(A,B, '.'); hold on;
axes('position', [0   0.2 0.2 0.8]); hist(B);
set(gca, 'xdir', 'reverse'); set(gca, 'view', [90 90]);
axes('position', [0.2 0   0.8 0.2]); hist(A);

figure
axes('position', [0.2 0.2 0.8 0.8]); plot(M1,M2, '.'); hold on;
axes('position', [0   0.2 0.2 0.8]); hist(M2);
set(gca, 'xdir', 'reverse'); set(gca, 'view', [90 90]);
axes('position', [0.2 0   0.8 0.2]); hist(M1);

%% plot the histograms showing the PDF of the signals.

figure; subplot(1,2,1)
[f1,xi1] = ksdensity(s);
hist(EEG.data(48,:),100);
hold on;
plotyy(nan,nan,xi1,f1)
 
subplot(1,2,2)
[f2,xi2] = ksdensity(activations(1,:));
hist(activations(1,:),100);
hold on
plotyy(nan,nan,xi2,f2)


data = [A;B]';
[bandwidth,density,X,Y]=kde2d(data,50);
figure
contour3(X,Y,density,50), hold on 
plot(data(:,1),data(:,2),'r.','MarkerSize',5)

figure
surf(X,Y,density,'LineStyle','none'), view([0,70]) 
colormap jet, hold on, alpha(.5) 
set(gca, 'color', 'white'); 
plot(data(:,1),data(:,2),'r.','MarkerSize',5)
