%% run_pf_24h.m — 24時間潮流計算（1h間隔 / 30min間隔の2ケース）
%  夏季ピーク日（8月平日）の日負荷曲線に沿って時系列潮流計算を実行
%  各時刻でNR法 → 不収束ならDC潮流にフォールバック
%
%  出力: output/matpower_alljapan/pf_24h_1h_results.mat
%        output/matpower_alljapan/pf_24h_30min_results.mat
%        output/matpower_alljapan/pf_24h_summary.png
clear; clc; close all;
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

%% Setup MATPOWER
MP = '/Users/shigenoburyuto/Documents/MATLAB/matpower8.1/matpower8.1';
addpath(fullfile(MP, 'lib'));
addpath(fullfile(MP, 'data'));
addpath(fullfile(MP, 'mips', 'lib'));
addpath(fullfile(MP, 'mp-opt-model', 'lib'));

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..');
out_dir = fullfile(project_root, 'output', 'matpower_alljapan');

%% Load data
ts = load(fullfile(out_dir, 'load_timeseries.mat'));
aj_data = load(fullfile(out_dir, 'alljapan.mat'));
aj_base = aj_data.mpc;
aj_base.bus(aj_base.bus(:, 13) == 0, 13) = 0.85;
base_pd = aj_base.bus(:, 3);
base_qd = aj_base.bus(:, 4);
base_pg = aj_base.gen(:, 2);

nb = size(aj_base.bus, 1);
ng = size(aj_base.gen, 1);
nbr = size(aj_base.branch, 1);

%% ========================================
%% Case 1: 24h @ 1h interval (24 steps)
%% ========================================
fprintf('=== 24h Power Flow @ 1h interval (24 steps) ===\n');
n_steps_1h = 24;
mults_1h = ts.ts_24h_1h_mult;
hours_1h = ts.ts_24h_1h_hour;

% Preallocate results
r1h.hours = hours_1h;
r1h.mult = mults_1h;
r1h.converged_ac = false(n_steps_1h, 1);
r1h.total_pd = zeros(n_steps_1h, 1);
r1h.total_pg = zeros(n_steps_1h, 1);
r1h.total_loss = zeros(n_steps_1h, 1);
r1h.vm_min = zeros(n_steps_1h, 1);
r1h.vm_max = zeros(n_steps_1h, 1);
r1h.vm_mean = zeros(n_steps_1h, 1);
r1h.va_max = zeros(n_steps_1h, 1);

mpopt_nr = mpoption('verbose', 0, 'out.all', 0, 'pf.nr.max_it', 20);
mpopt_dc = mpoption('verbose', 0, 'out.all', 0);
prev_vm = ones(nb, 1);
prev_va = zeros(nb, 1);

tic;
for t = 1:n_steps_1h
    m = mults_1h(t);
    mpc = aj_base;
    mpc.bus(:, 3) = base_pd * m;
    mpc.bus(:, 4) = base_qd * m;
    mpc.gen(:, 2) = base_pg * m;

    % Warm start from previous solution
    mpc.bus(:, 8) = prev_vm;
    mpc.bus(:, 9) = prev_va;

    % Try NR
    res = runpf(mpc, mpopt_nr);
    if res.success
        r1h.converged_ac(t) = true;
        prev_vm = res.bus(:, 8);
        prev_va = res.bus(:, 9);
    else
        res = rundcpf(mpc, mpopt_dc);
    end

    if res.success
        r1h.total_pd(t) = sum(res.bus(:, 3));
        r1h.total_pg(t) = sum(res.gen(:, 2));
        r1h.total_loss(t) = sum(res.branch(:, 14) + res.branch(:, 16));
        vm = res.bus(:, 8);
        r1h.vm_min(t) = min(vm);
        r1h.vm_max(t) = max(vm);
        r1h.vm_mean(t) = mean(vm);
        r1h.va_max(t) = max(abs(res.bus(:, 9)));
    end

    ac_str = 'AC'; if ~r1h.converged_ac(t), ac_str = 'DC'; end
    fprintf('  [%02d:00] mult=%.3f  Pd=%8.0f MW  %s\n', ...
        hours_1h(t), m, r1h.total_pd(t), ac_str);
end
elapsed_1h = toc;
fprintf('  Completed in %.1f sec (%d AC / %d DC)\n\n', ...
    elapsed_1h, sum(r1h.converged_ac), sum(~r1h.converged_ac));

save(fullfile(out_dir, 'pf_24h_1h_results.mat'), 'r1h', '-v7.3');

%% ========================================
%% Case 2: 24h @ 30min interval (48 steps)
%% ========================================
fprintf('=== 24h Power Flow @ 30min interval (48 steps) ===\n');
n_steps_30m = 48;
mults_30m = ts.ts_24h_30min_mult;
hours_30m = ts.ts_24h_30min_hour;

r30m.hours = hours_30m;
r30m.mult = mults_30m;
r30m.converged_ac = false(n_steps_30m, 1);
r30m.total_pd = zeros(n_steps_30m, 1);
r30m.total_pg = zeros(n_steps_30m, 1);
r30m.total_loss = zeros(n_steps_30m, 1);
r30m.vm_min = zeros(n_steps_30m, 1);
r30m.vm_max = zeros(n_steps_30m, 1);
r30m.vm_mean = zeros(n_steps_30m, 1);
r30m.va_max = zeros(n_steps_30m, 1);

prev_vm = ones(nb, 1);
prev_va = zeros(nb, 1);

tic;
for t = 1:n_steps_30m
    m = mults_30m(t);
    mpc = aj_base;
    mpc.bus(:, 3) = base_pd * m;
    mpc.bus(:, 4) = base_qd * m;
    mpc.gen(:, 2) = base_pg * m;
    mpc.bus(:, 8) = prev_vm;
    mpc.bus(:, 9) = prev_va;

    res = runpf(mpc, mpopt_nr);
    if res.success
        r30m.converged_ac(t) = true;
        prev_vm = res.bus(:, 8);
        prev_va = res.bus(:, 9);
    else
        res = rundcpf(mpc, mpopt_dc);
    end

    if res.success
        r30m.total_pd(t) = sum(res.bus(:, 3));
        r30m.total_pg(t) = sum(res.gen(:, 2));
        r30m.total_loss(t) = sum(res.branch(:, 14) + res.branch(:, 16));
        vm = res.bus(:, 8);
        r30m.vm_min(t) = min(vm);
        r30m.vm_max(t) = max(vm);
        r30m.vm_mean(t) = mean(vm);
        r30m.va_max(t) = max(abs(res.bus(:, 9)));
    end

    ac_str = 'AC'; if ~r30m.converged_ac(t), ac_str = 'DC'; end
    fprintf('  [%2d/%2d %05.1fh] mult=%.3f  Pd=%8.0f MW  %s\n', ...
        t, n_steps_30m, hours_30m(t), m, r30m.total_pd(t), ac_str);
end
elapsed_30m = toc;
fprintf('  Completed in %.1f sec (%d AC / %d DC)\n\n', ...
    elapsed_30m, sum(r30m.converged_ac), sum(~r30m.converged_ac));

save(fullfile(out_dir, 'pf_24h_30min_results.mat'), 'r30m', '-v7.3');

%% ========================================
%% Visualization
%% ========================================
fig = figure('Position', [50 50 1800 1000], 'Visible', 'off');

% Plot 1: Load curve comparison (1h vs 30min)
subplot(2, 3, 1);
plot(hours_1h, r1h.total_pd/1e3, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
plot(hours_30m, r30m.total_pd/1e3, 'r-', 'LineWidth', 1.5);
xlabel('時刻 [h]'); ylabel('総負荷 [GW]');
title('日負荷曲線'); legend('1h間隔', '30min間隔', 'Location', 'south');
grid on; xlim([0 23.5]);

% Plot 2: Generation tracking
subplot(2, 3, 2);
plot(hours_1h, r1h.total_pg/1e3, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
plot(hours_1h, r1h.total_pd/1e3, 'r--', 'LineWidth', 1.5);
plot(hours_1h, r1h.total_loss/1e3, 'k-', 'LineWidth', 1.5);
xlabel('時刻 [h]'); ylabel('電力 [GW]');
title('発電・負荷・損失'); legend('発電', '負荷', '損失', 'Location', 'south');
grid on; xlim([0 23]);

% Plot 3: Voltage envelope (1h)
subplot(2, 3, 3);
fill([hours_1h; flipud(hours_1h)], [r1h.vm_min; flipud(r1h.vm_max)], ...
    [0.8 0.9 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(hours_1h, r1h.vm_mean, 'b-', 'LineWidth', 2);
plot(hours_1h, r1h.vm_min, 'r-', 'LineWidth', 1);
plot(hours_1h, r1h.vm_max, 'r-', 'LineWidth', 1);
yline(1.0, 'k--');
xlabel('時刻 [h]'); ylabel('Vm [pu]');
title('電圧エンベロープ (1h)');
legend('Min-Max帯', '平均', 'Min/Max', 'Location', 'best');
grid on; xlim([0 23]);

% Plot 4: Voltage envelope (30min)
subplot(2, 3, 4);
fill([hours_30m; flipud(hours_30m)], [r30m.vm_min; flipud(r30m.vm_max)], ...
    [1.0 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;
plot(hours_30m, r30m.vm_mean, 'r-', 'LineWidth', 2);
plot(hours_30m, r30m.vm_min, 'b-', 'LineWidth', 1);
plot(hours_30m, r30m.vm_max, 'b-', 'LineWidth', 1);
yline(1.0, 'k--');
xlabel('時刻 [h]'); ylabel('Vm [pu]');
title('電圧エンベロープ (30min)');
grid on; xlim([0 23.5]);

% Plot 5: Loss curve
subplot(2, 3, 5);
yyaxis left;
plot(hours_1h, r1h.total_loss, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
ylabel('損失 [MW]');
yyaxis right;
loss_pct = r1h.total_loss ./ r1h.total_pg * 100;
loss_pct(r1h.total_pg == 0) = 0;
plot(hours_1h, loss_pct, 'r--', 'LineWidth', 1.5);
ylabel('損失率 [%]');
xlabel('時刻 [h]'); title('送電損失の時間変化');
grid on; xlim([0 23]);

% Plot 6: Load multiplier profile
subplot(2, 3, 6);
plot(hours_1h, mults_1h, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
plot(hours_30m, mults_30m, 'r-', 'LineWidth', 1.5);
yline(1.0, 'k--');
xlabel('時刻 [h]'); ylabel('負荷倍率');
title('日負荷曲線（正規化）');
legend('1h', '30min', 'Location', 'south');
grid on; xlim([0 23.5]);

sgtitle(sprintf('24時間潮流計算 — %d bus, %d branch (1h: %.1fs, 30min: %.1fs)', ...
    nb, nbr, elapsed_1h, elapsed_30m), 'FontSize', 14, 'FontWeight', 'bold');

png_path = fullfile(out_dir, 'pf_24h_summary.png');
exportgraphics(fig, png_path, 'Resolution', 150);
fprintf('Saved: %s\n', png_path);

fprintf('\n=== 24h Power Flow Complete ===\n');
fprintf('  1h interval:  %d steps, %.1f sec, %d/%d AC converged\n', ...
    n_steps_1h, elapsed_1h, sum(r1h.converged_ac), n_steps_1h);
fprintf('  30min interval: %d steps, %.1f sec, %d/%d AC converged\n', ...
    n_steps_30m, elapsed_30m, sum(r30m.converged_ac), n_steps_30m);
