close all; clear all; 


% %%
% %% System
% A = [1, -2.419, 1.939, -0.516];
% B = [0,0,.001,.00144, -.00681]; 
% sizey = size(A,1);
% %% Tuning parameters
% Wu =1;  % input weights
% Wy=1;   % output weights
% N=15;   % prediction horizon
% NU=3;   % control horizon
% 
% %% Set point
% 
% Yr = [zeros(1,200),-0.1*ones(1,200),0.1*ones(1,200),-0.1*ones(1,200),0.1*ones(1,200)];
% 
% 
% %% Closed-loop simulation
% 
% %%  Find prediction matrices  
% % yfut = H *Dufut + P*Dupast + Q*ypast
% 
% % Add integral to model
% sizey = size(A,1);
% D = [eye(sizey),-eye(sizey)];
% AD = conv(A,D);%convmat(A,D);
% nA = size(AD,2);
% nB = size(B,2);
% 
% %%%%  Initialise recursion data
% %%%% nominal model    y =  Bo ut + B2 Dupast  + A2 ypast
% A2 = -AD(1:sizey,sizey+1:nA);
% B2 = B(1:sizey,sizey+1:nB);
% Bo = B(1:sizey,1:sizey);
% nB2 = nB-sizey;
% nA2 = nA-sizey;
% 
% P1=Bo;
% P2=B2;
% P3=A2;
% 
% %%%%% Loop updating models using recursion
% for i=2:N
%    
%    vecold = (i-2)*sizey+1:(i-1)*sizey;
%    vecnew = (i-1)*sizey+1:i*sizey;
%    Phi = P3(vecold,1:sizey); 
% 
%    vecufut = 1:sizey*i;
%    vecufut2 = 1:sizey*(i-1);
%    P1(vecnew,vecufut) = [(P2(vecold,1:sizey)+Phi*Bo),P1(vecold,vecufut2)];
%    
%    vecupast = sizey+1:nB2;
%    vecypast = sizey+1:nA2;
%    
%    temp = [P2(vecold,vecupast),zeros(sizey,sizey)] + Phi*B2;
%    P2(vecnew,1:nB2) = temp;
%    P3(vecnew,1:nA2) = [P3(vecold,vecypast),zeros(sizey,sizey)] + Phi*A2;
% end
% 
% % figure(1)
% % plot(P2)
% 
% H=P1; 
% P=P2; 
% Q=P3;
%%




N=1000;
a = [1 -2.419 1.939 -0.516];
b = [0, 0, 0.01 0.00144 -0.00681];
theta = [2.419 -1.939 0.516 0.01 0.00144 -0.00681];

%Prediction matrixcies
horizon = 5;
A = conv(a, [1 -1]);
[Ca, Ha] = make_CaHa(horizon, A');
[Cb, Hb] = make_CaHa(horizon, b');
Cai = inv(Ca);
H = Cai * Cb; 
P = Cai * Hb;
disp(P)
Q = Cai * Ha;


y_future = [];
y = zeros(N,1);
%u = zeros(N,1);
y_pred = zeros(N,horizon);
sigma = 0.01;
time=[1 2 3 4 5]';
P_rls = eye(6); 
N_parameters = 200;
thetaest = zeros(N_parameters,6);
r = [zeros(200,1); -0.1*ones(200,1); 0.1*ones(200,1); -0.1*ones(200,1); 0.1*ones(200,1)];
u = 0.1*ones(N_parameters,1);

%initial prediction for model parameters
for t=6:N_parameters
    
    x = [y(t-1) y(t-2) y(t-3) u(t-3) u(t-4) u(t-5)]';
    y(t) = theta*x + normrnd(0, sigma);
    [thetaest(t,:), P_rls] = RLS(y(t), x, thetaest(t-1,:)', P_rls);
    time = [time; t];
end
theta_init = thetaest(end, :);
thetaest = ones(N,6) * diag(theta_init);
u = r; 
time=[1 2 3 4 5]';

for t=6:N
    
    x = [y(t-1) y(t-2) y(t-3) u(t-3) u(t-4) u(t-5)]';
    y(t) = theta*x + normrnd(0, sigma);
    [thetaest(t,:), P_rls] = RLS(y(t), x, thetaest(t-1,:)', P_rls);
    %Averaging paramter estimates over the last 6 samples for smoothnes 
    thetaest(t,:) = (1/6)*(thetaest(t-5:t,:)'*ones(6,1))';
    
    if t+horizon < N
        y_past = y(t-length(Q(1,:))+1:t);
        
        u_past = u(t-length(b)+1:t-1);
        delta_u_past = conv(u_past, [1 -1]);
        delta_u_past = delta_u_past(2:end);
        
        u_future = u(t:t+horizon-1);
        delta_u_future = conv(u_future, [1 -1]);
        delta_u_future = delta_u_future(2:end)
        
        y_future = H * delta_u_future + P * delta_u_past - Q * y_past; 
        y_pred(t, :) = y_future;
    end
    %u(t) = r(t);%(r(t)-y(t));
    time = [time; t];
end

figure(1) 
%disp(size(time));
plot(time, y);
hold on;
plot(time, r);
plot(time, y_pred(:,2), '-'); 
legend('y', 'reference', 'y pred n+2');

figure(2)
plot(time, thetaest(:,1));
hold on;
plot(time, thetaest(:,2));
plot(time, thetaest(:,3));
plot(time, thetaest(:,4));
plot(time, thetaest(:,5));
plot(time, thetaest(:,6));

legend('a1','a2', 'a3', 'b1', 'b2', 'b3');