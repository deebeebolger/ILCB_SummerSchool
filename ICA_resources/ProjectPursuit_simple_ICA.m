% Basic projection pursuit algorithm demonstrated on two sound signals.
% This code extracts one signal.

listen = 1;

%set random number seed.
seed = 99; 
rand('seed',seed); randn('seed',seed);

M = 2;   %The number of source signals.
N = 1e4; %The number of data points per signal. 

%% Load the sound data and set the variance of each source to unity. 
load chirp; s1 = y(1:N); s1=s1-mean(s1); s1=s1'/std(s1);
load train; s2 = y(1:N); s2=s2-mean(s2); s2=s2'/std(s2);

s = [s1; s2];  %Combine the sources...mixture
A = randn(M,M); % The mixing matrix

fs = 10000; %set the sampling rate.
soundsc(s(1,:),fs);
soundsc(s(2,:),fs);

audiowrite('/Users/bolger/Desktop/ILCBSummerSchool_2020/chirp_ext.wav',s1,fs);
audiowrite('/Users/bolger/Desktop/ILCBSummerSchool_2020/train_ext.wav',s2,fs);

%% Plot the histogram of each source signal (this approximates the pdf).
figure;
subplot(2,2,1); hist(s(1,:),50); title('histogram Chirp'); drawnow;
subplot(2,2,2); hist(s(2,:),50); title('histogram Train'); drawnow; 
subplot(2,2,3); plot(s(1,:),s(2,:),'.b'); title('source space: Chirp vs. Train'); drawnow;

%%

% figure; subplot(2,2,3)
% [f1,xi1] = ksdensity(M2);
% hist(B,50);
% hold on;
% plotyy(nan,nan,xi1,f1)


%% Create the mixtures x from M source signals s.
x = A*s;       %Taking the inner product

subplot(2,2,4); plot(x(1,:),x(2,:),'.r'); title('Mixture space'); drawnow; 

%% Listen to the mixture signals.
soundsc(x(1,:),fs);
soundsc(x(2,:),fs);

audiowrite('/Users/bolger/Desktop/ILCBSummerSchool_2020/chiptrain1_ext.wav',x(1,:),fs);
audiowrite('/Users/bolger/Desktop/ILCBSummerSchool_2020/chiptrain2_ext.wav',x(2,:),fs);
%% Sphere the mixtures using SVD
[U, D, V] = svd(x',0);
% set the new x to be left singular vectors of old x.
z = U;
%Each eigenvector has unit length, but we need to have unit variance
%mixtures.
z = z./repmat(std(z,1),N,1); 
z = z';

%% Initialize the unmixing vector to random vector.

W = eye(M,M); %Initialise to an identity matrix
%  w= w/norm(w); % Make sure that it has unit length. 

% Initialize the estimated source signal.
y = W*z; 

%Print out the initial correlations between each estimated source y and
%every source signal,s.
r=corrcoef([y;s]');
fprintf('Initial correlations of source and extracted signals\n');
% r1 = corrcoef([y; s1]');
% r2 = corrcoef([y; s2]');
rinitial = abs(r(M+1:2*M, 1:M));

%% 
maxiter = 100;    %Maximum number of iterations.
eta = 0.25;       %Step size for gradient ascent

%Make array hs to store values of function and gradient magnitude.
hs = zeros(maxiter,1);
gs = zeros(maxiter,1);

%% Begin gradient ascent on K...
writerObj = VideoWriter('/Users/bolger/Desktop/ILCBSummerSchool_2020/ICAtrainchirp2.avi'); % Name it.
writerObj.FrameRate = 60; % How many frames per second.
open(writerObj); 

figure('Color','white','WindowState','fullscreen');

for iter = 1:maxiter
    yout = W*z;             % Get estimated source signal, y.
    Y = tanh(yout);         %Get the estimated maximum entropy signal Y = cdf(y).
    detW = abs(det(W));     %Find value of function h
    h = ((1/N)*sum(sum(Y))+0.5*log(detW));
    %Find the matrix gradients @h/@W_ij
    g = inv(W')-(2/N)*Y*z';
    %Update W to increase h
    W=W+eta*g;
    %Record h and the magnitude of gradient
    hs(iter)=h;
    gs(iter)=norm(g(:));
    subplot(1,3,1)
    plot(yout(1,:),yout(2,:),'.c')
    set(gca,'XAxisLocation','origin','YAxisLocation','origin',...
        'XGrid','on','YGrid','on');
    drawnow
    subplot(1,3,2)
    plot(W(1,1)*z(1,:),W(1,2)*z(1,:),'.b')
    hold on
    plot(W(2,1)*z(2,:),W(2,2)*z(2,:),'.r')
      set(gca,'XAxisLocation','origin','YAxisLocation','origin',...
        'XGrid','on','YGrid','on');
    drawnow
    subplot(1,3,3)
    plot(hs,'*r');
    drawnow
    frame = getframe(gcf);
    writeVideo(writerObj, frame);
end

close(writerObj); % Saves the movie.

%% Plot the change in K and gradient/wopt angle during optimisation

figure;subplot(1,2,1);
plot(hs,'k');
title('Function values - Entropy');
xlabel('Iteration'); ylabel('h(Y)');
set(gca,'XLim',[0 maxiter])

subplot(1,2,2); plot(gs,'k');
title('Magnitude of entropy gradient');
xlabel('Iteraction'); ylabel('Gradient Magnitude');
set(gca,'XLim',[0 maxiter])

%% Print out the final correlations
r= corrcoef([y s]);
fprintf('Final correlations between source and extracted signals....\n');
rfinal = abs(r(M+1:2*M, 1:M))

%% Listen to the extracted signal

soundsc(yout(1,:),Fs);

soundsc(yout(2,:),Fs);

audiowrite('/Users/bolger/Desktop/chip-postica.wav',yout(1,:),fs);
audiowrite('/Users/bolger/Desktop/train-postica.wav',yout(2,:),fs);

