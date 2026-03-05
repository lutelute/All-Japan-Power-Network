%% run_pf_8760h.m — 年間8760時間潮流計算
%  日負荷曲線 × 月別係数で年間全時間の潮流計算を実行
%
%  モード:
%    mode = 'dc'  — DC潮流のみ（高速、約5分）
%    mode = 'ac'  — AC潮流（NR法）+ DCフォールバック（約2-5時間）
%
%  出力: output/matpower_alljapan/pf_8760h_results.mat
%        output/matpower_alljapan/pf_8760h_summary.png
clear; clc; close all;

%% Configuration
mode = 'dc';  % 'dc' or 'ac' — change to 'ac' for full AC analysis
save_interval = 500;  % Save intermediate results every N steps

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

N = 8760;
mults = ts.ts_8760_mult;
months = ts.ts_8760_month;
hours = ts.ts_8760_hour;
weekdays = ts.ts_8760_weekday;

fprintf('=== 8760h Power Flow (%s mode) ===\n', upper(mode));
fprintf('  %d buses, %d branches, %d gens\n', nb, nbr, ng);
fprintf('  Multiplier range: [%.3f, %.3f]\n\n', min(mults), max(mults));

%% Preallocate results
r.mode = mode;
r.hours = (0:N-1)';
r.mult = mults;
r.month = months;
r.hour_of_day = hours;
r.weekday = weekdays;
r.converged_ac = false(N, 1);
r.total_pd = zeros(N, 1);
r.total_pg = zeros(N, 1);
r.total_loss = zeros(N, 1);
r.vm_min = ones(N, 1);
r.vm_max = ones(N, 1);
r.vm_mean = ones(N, 1);
r.va_max = zeros(N, 1);
r.max_branch_flow = zeros(N, 1);

% Store key snapshots (peak, valley, spring, winter)
r.snapshots = struct();

mpopt_nr = mpoption('verbose', 0, 'out.all', 0, 'pf.nr.max_it', 100);
mpopt_dc = mpoption('verbose', 0, 'out.all', 0);

prev_vm = ones(nb, 1);
prev_va = zeros(nb, 1);

%% Main loop
tic;
n_ac = 0;
for t = 1:N
    m = mults(t);
    mpc = aj_base;
    mpc.bus(:, 3) = base_pd * m;
    mpc.bus(:, 4) = base_qd * m;
    mpc.gen(:, 2) = base_pg * m;

    if strcmp(mode, 'ac')
        % Warm start
        mpc.bus(:, 8) = prev_vm;
        mpc.bus(:, 9) = prev_va;

        res = runpf(mpc, mpopt_nr);
        if res.success
            r.converged_ac(t) = true;
            prev_vm = res.bus(:, 8);
            prev_va = res.bus(:, 9);
            n_ac = n_ac + 1;
        else
            res = rundcpf(mpc, mpopt_dc);
        end
    else
        % DC only
        res = rundcpf(mpc, mpopt_dc);
    end

    if res.success
        r.total_pd(t) = sum(res.bus(:, 3));
        r.total_pg(t) = sum(res.gen(:, 2));
        r.total_loss(t) = sum(res.branch(:, 14) + res.branch(:, 16));
        vm = res.bus(:, 8);
        r.vm_min(t) = min(vm);
        r.vm_max(t) = max(vm);
        r.vm_mean(t) = mean(vm);
        r.va_max(t) = max(abs(res.bus(:, 9)));
        Sf = abs(res.branch(:, 14) + 1j * res.branch(:, 15));
        r.max_branch_flow(t) = max(Sf);
    end

    % Save key snapshots
    [~, peak_idx] = max(mults(1:t));
    [~, valley_idx] = min(mults(1:t));
    if t == peak_idx
        r.snapshots.peak_hour = t;
        r.snapshots.peak_va = res.bus(:, 9);
    end
    if t == valley_idx
        r.snapshots.valley_hour = t;
        r.snapshots.valley_va = res.bus(:, 9);
    end

    % Progress
    if mod(t, 730) == 0  % ~monthly
        elapsed = toc;
        rate = t / elapsed;
        eta = (N - t) / rate;
        fprintf('  [%5d/%d] month=%2d  mult=%.3f  Pd=%8.0f MW  (%.0f steps/s, ETA %.0f s)\n', ...
            t, N, months(t), m, r.total_pd(t), rate, eta);
    end

    % Intermediate save
    if mod(t, save_interval) == 0
        save(fullfile(out_dir, 'pf_8760h_results_partial.mat'), 'r', 't', '-v7.3');
    end
end
elapsed_total = toc;

fprintf('\n  Completed: %d steps in %.1f sec (%.1f steps/s)\n', N, elapsed_total, N/elapsed_total);
if strcmp(mode, 'ac')
    fprintf('  AC converged: %d/%d (%.1f%%)\n', n_ac, N, n_ac/N*100);
end

%% Save final results
save(fullfile(out_dir, 'pf_8760h_results.mat'), 'r', '-v7.3');
fprintf('  Saved: pf_8760h_results.mat\n');

%% ========================================
%% Visualization
%% ========================================
fprintf('\n  Generating plots...\n');
fig = figure('Position', [50 50 1800 1200], 'Visible', 'off');

month_names = {'1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'};

% Plot 1: Annual load curve
subplot(3, 3, 1);
plot(r.hours/24, r.total_pd/1e3, '-', 'LineWidth', 0.3, 'Color', [0.2 0.5 0.9]);
xlabel('日数'); ylabel('負荷 [GW]');
title('年間負荷曲線 (8760h)'); grid on;
xlim([0 365]);

% Plot 2: Load duration curve
subplot(3, 3, 2);
pd_sorted = sort(r.total_pd, 'descend');
plot((1:N)/N*100, pd_sorted/1e3, 'r-', 'LineWidth', 2);
xlabel('時間率 [%]'); ylabel('負荷 [GW]');
title('負荷持続曲線'); grid on;

% Plot 3: Monthly box plot of load
subplot(3, 3, 3);
boxplot(r.total_pd/1e3, r.month);
set(gca, 'XTickLabel', month_names, 'XTickLabelRotation', 45);
ylabel('負荷 [GW]'); title('月別負荷分布'); grid on;

% Plot 4: Annual loss curve
subplot(3, 3, 4);
plot(r.hours/24, r.total_loss, '-', 'LineWidth', 0.3, 'Color', [0.9 0.4 0.2]);
xlabel('日数'); ylabel('損失 [MW]');
title('年間損失曲線'); grid on;
xlim([0 365]);

% Plot 5: Voltage envelope (only meaningful for AC mode)
subplot(3, 3, 5);
if strcmp(mode, 'ac') && any(r.converged_ac)
    ac_mask = r.converged_ac;
    ac_hours = r.hours(ac_mask);
    fill([ac_hours; flipud(ac_hours)]/24, ...
        [r.vm_min(ac_mask); flipud(r.vm_max(ac_mask))], ...
        [0.8 0.9 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    plot(ac_hours/24, r.vm_mean(ac_mask), 'b-', 'LineWidth', 0.5);
    xlabel('日数'); ylabel('Vm [pu]');
    title('電圧エンベロープ (AC収束時)');
else
    % DC mode: show voltage angles instead
    plot(r.hours/24, r.va_max, '-', 'LineWidth', 0.3, 'Color', [0.4 0.7 0.3]);
    xlabel('日数'); ylabel('最大位相角 [deg]');
    title('最大電圧位相角 (DC)');
end
grid on; xlim([0 365]);

% Plot 6: Max branch flow
subplot(3, 3, 6);
plot(r.hours/24, r.max_branch_flow, '-', 'LineWidth', 0.3, 'Color', [0.6 0.3 0.7]);
xlabel('日数'); ylabel('最大潮流 [MW]');
title('最大ブランチ潮流'); grid on;
xlim([0 365]);

% Plot 7: Monthly generation totals
subplot(3, 3, 7);
monthly_gen = zeros(12, 1);
monthly_loss = zeros(12, 1);
for m = 1:12
    mask = r.month == m;
    monthly_gen(m) = sum(r.total_pg(mask)) / 1e3;  % GWh
    monthly_loss(m) = sum(r.total_loss(mask)) / 1e3;
end
bar([monthly_gen, monthly_loss], 'stacked');
set(gca, 'XTickLabel', month_names, 'XTickLabelRotation', 45);
ylabel('電力量 [GWh]'); title('月別発電量・損失');
legend('発電', '損失', 'Location', 'northeast'); grid on;

% Plot 8: Weekday vs weekend pattern
subplot(3, 3, 8);
wd_mask = r.weekday == 1;
we_mask = r.weekday == 0;
% Average daily pattern
wd_avg = zeros(24, 1);
we_avg = zeros(24, 1);
for h = 0:23
    hm = r.hour_of_day == h;
    if any(hm & wd_mask)
        wd_avg(h+1) = mean(r.total_pd(hm & wd_mask));
    end
    if any(hm & we_mask)
        we_avg(h+1) = mean(r.total_pd(hm & we_mask));
    end
end
plot(0:23, wd_avg/1e3, 'b-o', 'LineWidth', 2);
hold on;
plot(0:23, we_avg/1e3, 'r-s', 'LineWidth', 2);
xlabel('時刻 [h]'); ylabel('平均負荷 [GW]');
title('平日 vs 休日パターン');
legend('平日', '休日', 'Location', 'south'); grid on;

% Plot 9: Summary stats
subplot(3, 3, 9);
axis off;
annual_energy = sum(r.total_pd) / 1e6;  % TWh
annual_loss = sum(r.total_loss) / 1e6;   % TWh
peak_pd = max(r.total_pd);
valley_pd = min(r.total_pd);
load_factor_ann = mean(r.total_pd) / peak_pd * 100;

txt = {
    sprintf('【年間サマリ】'),
    '',
    sprintf('  ピーク負荷: %.1f GW', peak_pd/1e3),
    sprintf('  最小負荷:   %.1f GW', valley_pd/1e3),
    sprintf('  年間負荷率: %.1f %%', load_factor_ann),
    sprintf('  年間電力量: %.1f TWh', annual_energy),
    sprintf('  年間損失量: %.2f TWh (%.1f%%)', annual_loss, annual_loss/annual_energy*100),
    '',
    sprintf('  モデル: %d bus, %d branch', nb, nbr),
    sprintf('  計算時間: %.1f sec (%.0f steps/s)', elapsed_total, N/elapsed_total),
    sprintf('  モード: %s', upper(mode)),
};
text(0.05, 0.95, txt, 'FontSize', 11, 'VerticalAlignment', 'top', ...
    'FontName', 'Helvetica', 'Interpreter', 'none');

sgtitle(sprintf('8760時間潮流計算 — %s mode', upper(mode)), ...
    'FontSize', 16, 'FontWeight', 'bold');

png_path = fullfile(out_dir, 'pf_8760h_summary.png');
exportgraphics(fig, png_path, 'Resolution', 150);
fprintf('  Saved: %s\n', png_path);

% Clean up partial save
partial_path = fullfile(out_dir, 'pf_8760h_results_partial.mat');
if isfile(partial_path), delete(partial_path); end

fprintf('\n=== 8760h Power Flow Complete ===\n');
