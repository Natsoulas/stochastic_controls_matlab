%stochastic adaptive controls project 1, 
% collaborated with Lorenzo Borrelli (lbohr)
%%
clc;clear;
v_u = 15;
v_w = 15;
Ts = 0.01; %sampling time
omega_u = 0;
theta_t = 120; %degrees
theta_w = -60;

%state space model for linear continuous-time
A = [0,0,-v_u*sind(theta_t);0,0,v_u*cosd(theta_t);0,0,0];%state transition
B = [cosd(theta_t),0;sind(theta_t),0;0,1];
C = diag([1,1,1]);
xi_t = normrnd(0,sqrt(0.5));
delta_t = normrnd(0,sqrt(10));

%stability check for continuous
[dontuse,diagonal] = eig(A);
%state space model for linear discrete-time
A_d = expm(Ts*A);
[t,B_d_column] = ode45(@ode_bd,[0,Ts],zeros(3,2));
B_d = [B_d_column(45,1:3)',B_d_column(45,4:6)'];
%stability check for discrete
[dontuse2, diagonald] = eig(A_d);
% Discrete system checking
sys = ss(A,B,C,zeros(3,2));
%h = nyquistplot(sys);
sys_discrete = c2d(sys,Ts);
R_meas = 0.2*eye(2);
%G_xi = [0;0;1];
G_cont = [cosd(theta_w),0;sind(theta_w),0;0,1]; %combined for noises
R_cont = diag([10,0.5]);
%struc_xi = expm([A,G_xi*R_xi*transpose(G_xi);zeros(3,3),-transpose(A)]*Ts);
%R_xi_tilde = struc_xi(1:3,4:6);
%R_xi_d = R_xi_tilde*transpose(A_d);

%G_delta = [cosd(theta_w);sind(theta_w);0];
%struc_delta = expm([A,G_delta*R_delta*transpose(G_delta);zeros(3,3),-transpose(A)]*Ts);
%R_delta_tilde = struc_delta(1:3,4:6);

%R_delta_d = R_delta_tilde*transpose(A_d);
aux = [A,G_cont*R_cont*G_cont';zeros(3,3),-A'];
struc_trick = expm(aux*Ts);
R_tilde = struc_trick(1:3,4:6);
R_d = R_tilde*A_d';
R_d(abs(R_d)<1e-3)=0;
%continous system process noise covariance matrix to attempt to solve lyapunov
%equation for stationary distribution
%G_combined = [cosd(theta_w),cosd(theta_w),0;sind(theta_w),sind(theta_w),0;0,0,1];
%nonlinear_covariance = diag([10,0.5]);
%Q_contin = G_combined*nonlinear_covariance*G_combined'; %continuous-time process noise covariance matrix 
%stationary_cov_continuous = lyap(A,-Q_contin); %could not be solved
%
%discrete system process noise covariance matrix to attempt to solve
%lyapunov equation for stationary distribution
%Q_d = R_xi_d + R_delta_d;
%stationary_cov_discrete = dlyap(A_d,-Q_d);
%
%[t_other, xi_k_series] = ode45(@(t,y) ode_xi_k(t,y,A,G_xi,xi_t,Ts),[0 Ts],zeros(3,1));
%[t_dother, delta_k_series] = ode45(@(t,y) ode_delta_k(t,y,A,G_delta,delta_t,Ts),[0 Ts],zeros(3,1));
%xi_k = xi_k_series(end,:)';
%delta_k = delta_k_series(end,:)';

%LQR DESIGN

Qz = diag([10,10,100]);
Qu = diag([0.02,0.1]);
N = 100;%placeholder for real N value
[L,S,e] = dlqr(sys_discrete.A,sys_discrete.B,Qz,Qu);

%1.4 simulations
A_cl_d = (A_d - B_d*L);
stationary_cov_cl = dlyap(A_cl_d,R_d); %thank god it actually worked,solved for stationary covariance matrix
%this means diagonals are the variances of each state vector element and
%means are zero just cuz
%for simulation, 95% confidence interval is being within 2 standard
%deviations from the mean
num_sims = 100;
num_steps = 1001;
%variances per state term
Rx_stationary = stationary_cov_cl(1,1);%R_delta_d(1,1);
Ry_stationary = stationary_cov_cl(2,2);%R_delta_d(2,2);
Rtheta_stationary = stationary_cov_cl(3,3);%R_xi_d(3,3);
sigma_x_stationary = sqrt(Rx_stationary);
sigma_y_stationary = sqrt(Ry_stationary);
sigma_theta_stationary = sqrt(Rtheta_stationary);
Hache = [1,0,0;0,1,0];
cov_output_stat_14 = Hache*stationary_cov_cl*Hache' + R_meas;
sigma_xout_stationary = sqrt(cov_output_stat_14(1,1));
cov_ctrl_stat_14 = L*stationary_cov_cl*L';
sigma_vu_control14 = sqrt(cov_ctrl_stat_14(1,1));
%std deviations (sigma)
lchol = chol(R_d,"lower");

%sigma_theta = sqrt(Rtheta);
save_x = zeros(num_steps,num_sims);
save_y = zeros(num_steps,num_sims);
save_theta = zeros(num_steps,num_sims);
save_state = zeros(num_steps,num_sims,3);
save_control = zeros(num_steps,num_sims,2);
save_output = zeros(num_steps,num_sims,2);
for sim = 1:1:num_sims
    z0 = [normrnd(0,1);normrnd(0,1);normrnd(0,0.5)];
    zk = z0;
    H = [1,0,0;0,1,0];
    for step = 1:1:num_steps 
        noise_k = lchol*randn(3,1);
        meas_noise_k = chol(R_meas)*randn(2,1);
        z_kplus1 = A_cl_d*zk + noise_k;
        save_x(step,sim) = zk(1,1);
        save_y(step,sim) = zk(2,1);
        save_theta(step,sim) = zk(3,1);

        save_state(step,sim,:) = zk;
        save_control(step,sim,:) = -L*zk;
        save_output(step,sim,:) = H*zk+meas_noise_k;

        zk = z_kplus1;
    end
end
time_steps = linspace(0,num_steps*Ts,num_steps);

figure
subplot(3,1,1);
plot(time_steps,save_x)
yline(2*sigma_x_stationary,"LineWidth",1,"Color","r")
yline(-2*sigma_x_stationary,"LineWidth",1,"Color","r")
xlabel("time (seconds)")
ylabel("x")

subplot(3,1,2); 
plot(time_steps,save_y)
yline(2*sigma_y_stationary,"LineWidth",1,"Color","r")
yline(-2*sigma_y_stationary,"LineWidth",1,"Color","r")
xlabel("time (seconds)")
ylabel("y")

subplot(3,1,3);
plot(time_steps,save_theta)
yline(2*sigma_theta_stationary,"LineWidth",1,"Color","r")
yline(-2*sigma_theta_stationary,"LineWidth",1,"Color","r")
xlabel("time (seconds)")
ylabel("theta")

figure
subplot(3,1,1)
plot(linspace(1,num_steps,num_steps),save_state(:,:,1))
yline(2*sigma_x_stationary,"LineWidth",1.5,"Color","r")
yline(-2*sigma_x_stationary,"LineWidth",1.5,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("x - state")
subplot(3,1,2)
plot(linspace(1,num_steps,num_steps),save_output(:,:,1))
yline(2*sigma_xout_stationary,"LineWidth",1.5,"Color","r")
yline(-2*sigma_xout_stationary,"LineWidth",1.5,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("x - output")
subplot(3,1,3)
plot(linspace(1,num_steps,num_steps),save_control(:,:,1))
yline(2*sigma_vu_control14,"LineWidth",1.5,"Color","r")
yline(-2*sigma_vu_control14,"LineWidth",1.5,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("v_u - control signal")


%sigma_vu_control14
%%%ORDINARY KALMAN FILTER CALL
%%
H = [1,0,0;0,1,0];
R_meas = 0.2*eye(2);
z0_kf = [50;-10;90*pi/180];
P0_kf = [10, 5, 0; 5, 20, 0; 0,0,30];
steps_kf = 1000;
%
[x_est,y_est,x_true,y_true,res] = ordinaryKF(A_d,H,R_d,R_meas,z0_kf,P0_kf,steps_kf);
%plot results
full_kf_steps = linspace(1,steps_kf+1,steps_kf+1)';
figure
plot(full_kf_steps,x_true,full_kf_steps,x_est)
legend("true","estimate")

est_err = [x_true';y_true'] - [x_est';y_est'];
% theoretical stationary covariance (1.5)
[P_p_inf,dnotuse,notuse] = idare(A_d,[1,0,0;0,1,0;0,0,0],R_d,[0.2,0,0;0,0.2,0;0,0,0],[],[]);
P_o_inf_inv = H'*inv(R_meas)*H; % + inv(P_p_inf)
P_o_inf = inv(P_o_inf_inv);
sigma_stationary_kf_x = sqrt(P_o_inf(1,1));

figure
plot(full_kf_steps',est_err(1,:))
yline(2*sigma_stationary_kf_x,"LineWidth",1,"Color","r");
yline(-2*sigma_stationary_kf_x,"LineWidth",1,"Color","r");
ylabel('error')
xlabel('steps')

%% Question 1.6

%1 Residuals: The residuals are the differences between the predicted measurements and the actual measurements.

% If residuals are normally distributed with zero mean and constant variance, the assumptions are appropriate.

% if residuals exhibit patterns or have non-constant variance, it indicate that the model assumptions are not met.

figure
subplot(2,1,1)

plot((res(:,1)))

yline(mean(res(:,1)))

subplot(2,1,2)

plot((res(:,2)))

yline(mean(res(:,2)))

var(res(:,1))

var(res(:,2))

% as a result mean 0 and almost constant variance on both 0.2035 and 0.2049

 

%2 Innovations are the differences between the predicted state and the actual state of the system.

% If the innovations are normally distributed with zero mean and constant variance, the model assumptions are appropriate.

%eps=y_meas-H

%3 ACF should be white noise

autocorr(res(:,1))

autocorr(res(:,2))

%%
%%% LQG CALL
steps_lqg = 1000;

[x_est_lqg,y_est_lqg,x_true_lqg,y_true_lqg,u_lqg_out] = LQG_separate(A_d,H,B_d,L,R_d,R_meas,z0_kf,P0_kf,steps_lqg); %A,H,B,L,R_sys,R_meas,z0,P0,steps
%plot results
full_lqg_steps = linspace(1,steps_lqg+1,steps_lqg+1)';
figure
plot(full_lqg_steps,x_true_lqg,full_lqg_steps,x_est_lqg)
legend("true","estimate")

est_err_lqg = [x_true_lqg';y_true_lqg'] - [x_est_lqg';y_est_lqg'];


figure
plot(full_lqg_steps',est_err_lqg(1,:),full_lqg_steps',est_err_lqg(2,:))
ylabel('error')
xlabel('steps')

figure
plot(full_lqg_steps(1:end-1,1),u_lqg_out(:,1))
ylabel('control output')
xlabel('steps')
%%
% %1.8 dependent noise model
% xidep_0 = [0;0;0];
% steps_w = 10000000;
% R1eta = [1,0,0;0.2,0.8,0;0.1,0,0.7];
% w_dep = dependent_noise(R_meas,R1eta,xidep_0,steps_w);
% w_step_series = linspace(1,steps_w,steps_w);
% figure
% plot(w_step_series,w_dep(:,1))
% % we must first find the stationary covariance matrix for the new noise
% % term w
% data_stat = w_dep';
% % Calculate the mean of each row
% M_stat = mean(data_stat, 2);
% 
% % Subtract the mean from each column
% Y_stat = data_stat - M_stat;
% 
% % Calculate the covariance matrix
% Q_newnoise = (1 / (size(data_stat, 2) - 1)) * (Y_stat * Y_stat');
%save this calc since it takes a while:
Q_newnoise = [1.00015578162269,0.360143406210871,0.169876393991212;0.360143406210871,0.615619290544161,0.0499668259321243;0.169876393991212,0.0499668259321243,0.459706162441587];

%% Q 1.9 LQR design
%find new LQR gain L_new
% L_new should be the same as L from 1.4 as the stationary LQR design 
% does not depend on the 
[L_new,S_new,e_new] = dlqr(A_d,B_d,Qz,Qu);
%find the closed-loop description
A_cl_d_new = (A_d-B_d*L_new);
%find theoretical stationary distributions of the states, output, and
%control signal
state_theor_1_9 = dlyap(A_cl_d_new,Q_newnoise); %stationary covariance of states
control_theor_1_9 = L_new*state_theor_1_9*L_new';
output_theor_1_9 = H*state_theor_1_9*H' + R_meas;
%verify the distributions experimentally
num_steps_19 = 1000;
num_sims_19 = 100;
[state_19, output_19, control_19] = simulate_cl_19(A_d,B_d,L_new,H,Q_newnoise,R_meas,num_steps_19,num_sims_19);
    %plot 1st element in state, control and output to verify theoretical
    %covariances
sigma_x_state19 = sqrt(state_theor_1_9(1,1));
sigma_x_output19 = sqrt(output_theor_1_9(1,1));
sigma_vu_control19 = sqrt(control_theor_1_9(1,1));

figure
subplot(3,1,1);
plot(linspace(1,num_steps_19,num_steps_19), state_19(:,:,1))
yline(2*sigma_x_state19,"LineWidth",3,"Color","r")
yline(-2*sigma_x_state19,"LineWidth",3,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("x - state")
subplot(3,1,2);
plot(linspace(1,num_steps_19,num_steps_19), output_19(:,:,1))
yline(2*sigma_x_output19,"LineWidth",3,"Color","r")
yline(-2*sigma_x_output19,"LineWidth",3,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("x - output")
subplot(3,1,3);
plot(linspace(1,num_steps_19,num_steps_19), control_19(:,:,1))
yline(2*sigma_vu_control19,"LineWidth",3,"Color","r")
yline(-2*sigma_vu_control19,"LineWidth",3,"Color","r")
xlabel("steps (Ts = 0.01)")
ylabel("v_u - control signal boatspeed")
%% Q 1.10 Comparison of New Variances of outputs and control signals for 1.4 and 1.9 cl systems
% I indeed do observe a big difference in the new variances and the old 
% variances. The new variances of the outputs and control signals are much 
% larger in 1.9 compared to those observed in 1.4. This can be seen by
% putting figures 2 and 9 side by side and reading their y-axes to visually
% see the factor of 2 to 5 difference by which the  1.9 results vary much
% more largely.

%% Q 1.11: Ordinary KF implementation hardware limitations?
% There are many hardware limitations that exist in implementing an ordinary 
% kalman filter. For one, the memory on whatever computer or chip you are 
% running it on is a limitation as the KF needs to store a bunch of 
% matrices and values each time it make an estimate. This is especially 
% true when dealing with large state spaces and measurement spaces. Another
% limitation is also computational with the computer processor needing to
% be fast enough to allow for effective estimation via performing really
% fast linear algebra operations. Measurement noise can also be a
% limitation as the ordinary kalman filter algorithm is for gaussian white
% noise. Sampling time also needs to be relatively small enough for the
% estimation to work. Lets say if measurements were provided every 1 hour,
% that might be too slow for a boat...

%% Q 1.12 Design a stationary predictive KF, find K_inf, what is cov of correlated noise signals?

%%

function Bd_s_column = ode_bd(t,~)
    v_u = 15;
    v_w = 15;
    Ts = 0.01; %sampling time
    omega_u = 0;
    theta_t = 120; %degrees
    theta_w = -60;
    A = [0,0,-v_u*sind(theta_t);0,0,v_u*cosd(theta_t);0,0,0];%state transition
    B = [cosd(theta_t),0;sind(theta_t),0;0,1];
    bd_square = expm(A*(Ts-t))*B;
    Bd_s_column = [bd_square(:,1); bd_square(:,2)];
end
function xi_k_s_column = ode_xi_k(t,~,A,G_xi,xi_t,Ts)
    xi_k_s_square = expm(A*(Ts-t))*G_xi*xi_t;
    xi_k_s_column = xi_k_s_square;
end
function delta_k_s_column = ode_delta_k(t,~,A,G_delta,delta_t,Ts)
    delta_k_s_square = expm(A*(Ts-t))*G_delta*delta_t;
    delta_k_s_column = delta_k_s_square;
end
function [x_est,y_est,x_true,y_true,res] = ordinaryKF(A,H,R_sys,R_meas,z0,P0,steps)
    %A should be Ad
    %R_sys should b e Rd
    zkmin1 = z0;
    Pkmin1 = P0;
    z_t_kmin1 = z0;
    x_est = zeros(steps+1,1);
    y_est = zeros(steps+1,1);
    x_true = zeros(steps+1,1);
    y_true = zeros(steps+1,1);
    x_est(1,1) = z0(1,1);
    y_est(1,1) = z0(2,1);
    x_true(1,1) = z0(1,1);
    y_true(1,1) = z0(2,1);
    res = zeros(steps,2);
    for step = 1:1:steps
        %truth
        syschol = chol(R_sys,"lower");
        sys_noise = syschol*randn(3,1);
        Auxilary = A*z_t_kmin1 + sys_noise;
        y_true_k = H*(Auxilary);
        %measurement
        measchol = chol(R_meas,"lower");
        meas_noise = measchol*randn(2,1);
        y_meas_k = y_true_k + meas_noise;
        res(step,:) = meas_noise';
        %prediction step
        z_p_k = A*zkmin1;
        P_p_k = A*Pkmin1*A' + R_sys;
        %update step`
        K_k = P_p_k*H'*inv(H*P_p_k*H'+ R_meas);
        zk = z_p_k + K_k*(y_meas_k - H*z_p_k);
        P_k = (eye(3)-K_k*H)*P_p_k;
        %save it all off
        x_est(1+step,1) = zk(1,1);
        y_est(1+step,1) = zk(2,1);
        x_true(1+step,1) = y_true_k(1,1);
        y_true(1+step,1) = y_true_k(2,1);
        %increment state and state covariance for next iteration
        zkmin1 = zk;
        Pkmin1 = P_k;
        z_t_kmin1 = Auxilary;
    end
end
function [x_est,y_est,x_true,y_true,u_lqg] = LQG_separate(A,H,B,L,R_sys,R_meas,z0,P0,steps)
    %A should be Ad
    %R_sys should b e Rd
    zkmin1 = z0;
    Pkmin1 = P0;
    z_t_kmin1 = z0;
    x_est = zeros(steps+1,1);
    y_est = zeros(steps+1,1);
    x_true = zeros(steps+1,1);
    y_true = zeros(steps+1,1);
    x_est(1,1) = z0(1,1);
    y_est(1,1) = z0(2,1);
    x_true(1,1) = z0(1,1);
    y_true(1,1) = z0(2,1);
    u_lqg = zeros(steps,3);
    for step = 1:1:steps
        %truth
        syschol = chol(R_sys,"lower");
        sys_noise = syschol*randn(3,1);
        Auxilary = A*z_t_kmin1 -B*L*[x_est(step,1);y_est(step,1);zkmin1(3,1)]+ sys_noise;
        y_true_k = H*(Auxilary);
        %measurement
        measchol = chol(R_meas,"lower");
        meas_noise = measchol*randn(2,1);
        %y_meas_k = y_true_k + meas_noise;
        %prediction step
        %z_p_k = A*zkmin1;
        P_p_k = A*Pkmin1*A' + R_sys;
        %update step`
        K_k = P_p_k*H'*inv(H*P_p_k*H'+ R_meas);
        %zk = z_p_k + K_k*(y_meas_k - H*z_p_k); %from KF (not LQG)
        u_lqg(step,:) = (-B*L*zkmin1)';
        zk = K_k*H*[y_true_k; Auxilary(3,1)] + (A-K_k*H-B*L)*zkmin1 -K_k*meas_noise;
        P_k = (eye(3)-K_k*H)*P_p_k;
        %save it all off
        x_est(1+step,1) = zk(1,1);
        y_est(1+step,1) = zk(2,1);
        x_true(1+step,1) = y_true_k(1,1);
        y_true(1+step,1) = y_true_k(2,1);
        %increment state and state covariance for next iteration
        zkmin1 = zk;
        Pkmin1 = P_k;
        z_t_kmin1 = Auxilary;
    end
end

function w_dep = dependent_noise(R_meas,R1eta,xi_0,steps)
    w_dep = zeros(steps,3);
    xi_k = xi_0;
    for step = 1:1:steps
        e_k = chol(R_meas,"lower")*randn(2,1);
        w_k = [1,0,0;0.2,0.8,0;0.1,0,0.7]*xi_k + [0,0;0,0;0.01,-0.01]*e_k;
        w_dep(step,:) = w_k';
        eta_k = chol(R1eta,"lower")*randn(3,1);
        xi_kplus1 = [0,0,0;0,0,0;0,0,6/13]*xi_k + eta_k;
        xi_k = xi_kplus1;
    end
end

function [w_k,e_k,xi_k] = noises(R_meas,R1eta, xi_k)
    e_k = chol(R_meas,"lower")*randn(2,1);
    w_k = [1,0,0;0.2,0.8,0;0.1,0,0.7]*xi_k + [0,0;0,0;0.01,-0.01]*e_k;
    eta_k = chol(R1eta,"lower")*randn(3,1);
    xi_kplus1 = [0,0,0;0,0,0;0,0,6/13]*xi_k + eta_k;
    xi_k = xi_kplus1;
end

function [state, output, control] = simulate_cl_19(A,B,L,H,R_sys,R_meas,num_steps,num_sims)

%sigma_theta = sqrt(Rtheta);
state = zeros(num_steps,num_sims,3);
output = zeros(num_steps,num_sims,2);
control = zeros(num_steps,num_sims,2);
R_sys = R_sys;
R1eta = [1,0,0;0.2,0.8,0;0.1,0,0.7];
for sim = 1:1:num_sims
    z0 = [normrnd(0,1);normrnd(0,1);normrnd(0,0.5)];
    zk = z0;
    xi_k = zeros(3,1);
    for step = 1:1:num_steps 
        %noise_k = chol(R_sys,"lower")*randn(3,1);
        %measnoise_k = chol(R_meas,"lower")*randn(2,1);
        [w_k,e_k,xi_k] = noises(R_meas,R1eta, xi_k);
        u_k = -L*zk;
        z_kplus1 = A*zk + B*u_k + w_k;%noise_k;
        state(step,sim,:) = zk;
        output(step,sim,:) = H*zk + e_k; %measnoise_k;
        control(step,sim,:) = u_k;
        zk = z_kplus1;
    end
end
end
