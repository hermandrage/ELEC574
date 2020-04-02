close all; clear all; 


N=1000;
a = [1 -2.419 1.939 -0.516];
b = [0, 0, 0.01 0.00144 -0.00681];
theta = [2.419 -1.939 0.516 0.01 0.00144 -0.00681];

%Prediction matrixcies
prediction_horizon = 5;
A = conv(a, [1 -1]);
[Ca, Ha] = make_CaHa(prediction_horizon, A');
[Cb, Hb] = make_CaHa(prediction_horizon, b');
Cai = inv(Ca);
H = Cai * Cb; 
P = Cai * Hb;
Q = Cai * Ha;


y_future = [];
y = zeros(N,1);
%u = zeros(N,1);
y_pred = zeros(N,prediction_horizon);
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
    
    %updating H, P and Q matix for prediction
    a = [1 -1*thetaest(t,1:3)];
    b = [0 0 thetaest(t, 4:end)];
    A = conv(a, [1 -1]);
    [Ca, Ha] = make_CaHa(prediction_horizon, A');
    [Cb, Hb] = make_CaHa(prediction_horizon, b');
    Cai = inv(Ca);
    H = Cai * Cb; 
    P = Cai * Hb;
    Q = Cai * Ha; 
    
    %making prediction
    if t+prediction_horizon < N
        y_past = y(t-length(Q(1,:))+1:t);
        
        u_past = u(t-length(b)+1:t-1);
        delta_u_past = conv(u_past, [1 -1]);
        delta_u_past = delta_u_past(2:end);
        
        u_future = u(t:t+prediction_horizon-1);
        delta_u_future = conv(u_future, [1 -1]);
        delta_u_future = delta_u_future(2:end);
        
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
plot(time, y_pred(:,3), '-'); 
legend('y', 'reference', 'y pred n+3');

figure(2)
plot(time, thetaest(:,1));
hold on;
plot(time, thetaest(:,2));
plot(time, thetaest(:,3));
plot(time, thetaest(:,4));
plot(time, thetaest(:,5));
plot(time, thetaest(:,6));

legend('a1','a2', 'a3', 'b1', 'b2', 'b3');