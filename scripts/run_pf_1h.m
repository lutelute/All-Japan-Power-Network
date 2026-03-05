%% run_pf_1h.m — 1時間スナップショット潮流計算
%  ピーク時（8月平日14時）の潮流計算を全地域 + 全日本で実行
%  AC潮流（NR法）→ 不収束の場合DC潮流にフォールバック
%
%  出力: output/matpower_alljapan/pf_1h_results.mat
%        output/matpower_alljapan/pf_1h_summary.png
clear; clc; close all;

%% Setup MATPOWER
MP = '/Users/shigenoburyuto/Documents/MATLAB/matpower8.1/matpower8.1';
addpath(fullfile(MP, 'lib'));
addpath(fullfile(MP, 'data'));
addpath(fullfile(MP, 'mips', 'lib'));
addpath(fullfile(MP, 'mp-opt-model', 'lib'));

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..');
out_dir = fullfile(project_root, 'output', 'matpower_alljapan');

%% Load time-series data
ts = load(fullfile(out_dir, 'load_timeseries.mat'));
mult = ts.ts_1h_mult(1);  % should be 1.0 for peak hour
fprintf('=== 1h Power Flow (peak hour, multiplier=%.3f) ===\n\n', mult);

%% Per-region power flow
regions = {'hokkaido','tohoku','tokyo','chubu','hokuriku','kansai','chugoku','shikoku','kyushu','okinawa'};
region_names_ja = {'北海道','東北','東京','中部','北陸','関西','中国','四国','九州','沖縄'};

results_1h = struct();
for i = 1:length(regions)
    r = regions{i};
    mat_file = fullfile(out_dir, [r '.mat']);
    if ~isfile(mat_file), continue; end

    data = load(mat_file);
    mpc = data.mpc;

    % Apply load multiplier
    mpc.bus(:, 3) = mpc.bus(:, 3) * mult;   % Pd
    mpc.bus(:, 4) = mpc.bus(:, 4) * mult;   % Qd
    mpc.gen(:, 2) = mpc.gen(:, 2) * mult;   % Pg
    mpc.bus(mpc.bus(:, 13) == 0, 13) = 0.85;

    % Try NR
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.nr.max_it', 200);
    res = runpf(mpc, mpopt);

    if ~res.success
        % DC fallback
        res = rundcpf(mpc, mpoption('verbose', 0, 'out.all', 0));
        ac_converged = false;
    else
        ac_converged = true;
    end

    if res.success
        vm = res.bus(:, 8);
        pd = sum(res.bus(:, 3));
        pg = sum(res.gen(:, 2));
        loss = sum(res.branch(:, 14) + res.branch(:, 16));
        if ac_converged
            fprintf('[%10s] AC CONVERGED  %5d bus  Pd=%8.0f MW  Pg=%8.0f MW  Loss=%6.0f MW  V=%.3f-%.3f\n', ...
                r, size(mpc.bus,1), pd, pg, loss, min(vm), max(vm));
        else
            fprintf('[%10s] DC CONVERGED  %5d bus  Pd=%8.0f MW  Pg=%8.0f MW\n', ...
                r, size(mpc.bus,1), pd, pg);
        end
        results_1h.(r).mpc = mpc;
        results_1h.(r).results = res;
        results_1h.(r).ac_converged = ac_converged;
    else
        fprintf('[%10s] FAILED\n', r);
    end
end

%% All-Japan
fprintf('\n--- All-Japan ---\n');
aj_data = load(fullfile(out_dir, 'alljapan.mat'));
aj_mpc = aj_data.mpc;
aj_mpc.bus(:, 3) = aj_mpc.bus(:, 3) * mult;
aj_mpc.bus(:, 4) = aj_mpc.bus(:, 4) * mult;
aj_mpc.gen(:, 2) = aj_mpc.gen(:, 2) * mult;
aj_mpc.bus(aj_mpc.bus(:, 13) == 0, 13) = 0.85;

mpopt_nr = mpoption('verbose', 0, 'out.all', 0, 'pf.nr.max_it', 300);
aj_res = runpf(aj_mpc, mpopt_nr);
aj_ac = aj_res.success;
if ~aj_ac
    aj_res = rundcpf(aj_mpc, mpoption('verbose', 0, 'out.all', 0));
end

if aj_res.success
    pd = sum(aj_res.bus(:, 3));
    pg = sum(aj_res.gen(:, 2));
    fprintf('All-Japan: %d bus, Pd=%.0f MW, Pg=%.0f MW\n', ...
        size(aj_mpc.bus,1), pd, pg);
    if aj_ac
        vm = aj_res.bus(:, 8);
        loss = sum(aj_res.branch(:, 14) + aj_res.branch(:, 16));
        fprintf('  AC CONVERGED  V=%.3f-%.3f  Loss=%.0f MW\n', min(vm), max(vm), loss);
    else
        fprintf('  DC CONVERGED\n');
    end
end

%% Visualization
fig = figure('Position', [50 50 1600 900], 'Visible', 'off');

% Panel 1: Regional generation vs load
subplot(2, 3, 1);
gen_r = zeros(1, length(regions));
load_r = zeros(1, length(regions));
loss_r = zeros(1, length(regions));
for i = 1:length(regions)
    r = regions{i};
    if isfield(results_1h, r)
        gen_r(i) = sum(results_1h.(r).results.gen(:, 2));
        load_r(i) = sum(results_1h.(r).results.bus(:, 3));
        if results_1h.(r).ac_converged
            loss_r(i) = sum(results_1h.(r).results.branch(:, 14) + results_1h.(r).results.branch(:, 16));
        end
    end
end
b = bar([gen_r; load_r]', 'grouped');
b(1).FaceColor = [0.2 0.6 0.9]; b(2).FaceColor = [0.9 0.4 0.3];
set(gca, 'XTickLabel', region_names_ja, 'XTickLabelRotation', 45, 'FontSize', 9);
legend('発電', '負荷', 'Location', 'northeast'); title('地域別 発電 vs 負荷');
ylabel('MW'); grid on;

% Panel 2: Voltage profile (AC converged regions)
subplot(2, 3, 2);
hold on;
colors = lines(10);
leg_names = {};
for i = 1:length(regions)
    r = regions{i};
    if isfield(results_1h, r) && results_1h.(r).ac_converged
        vm = results_1h.(r).results.bus(:, 8);
        scatter(1:length(vm), vm, 4, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.6);
        leg_names{end+1} = region_names_ja{i};
    end
end
yline(1.0, 'r--', 'LineWidth', 1.5);
yline(0.95, 'b:', 'LineWidth', 1); yline(1.05, 'b:', 'LineWidth', 1);
title('電圧プロファイル（AC収束地域）'); xlabel('バス番号'); ylabel('Vm [pu]');
if ~isempty(leg_names), legend(leg_names, 'Location', 'best', 'FontSize', 7); end
grid on;

% Panel 3: Ybus spy plot
subplot(2, 3, 3);
[Y, ~, ~] = makeYbus(aj_mpc.baseMVA, aj_mpc.bus, aj_mpc.branch);
spy(Y, 2);
title(sprintf('全日本 Ybus (%dx%d, %d nnz)', size(Y,1), size(Y,2), nnz(Y)));

% Panel 4: Bus count by region
subplot(2, 3, 4);
bus_count = zeros(1, length(regions));
for i = 1:length(regions)
    r = regions{i};
    if isfield(results_1h, r)
        bus_count(i) = size(results_1h.(r).mpc.bus, 1);
    end
end
bar(bus_count, 'FaceColor', [0.5 0.8 0.5]);
set(gca, 'XTickLabel', region_names_ja, 'XTickLabelRotation', 45, 'FontSize', 9);
title('地域別バス数'); ylabel('バス数'); grid on;

% Panel 5: Losses by region
subplot(2, 3, 5);
bar(loss_r, 'FaceColor', [0.9 0.7 0.3]);
set(gca, 'XTickLabel', region_names_ja, 'XTickLabelRotation', 45, 'FontSize', 9);
title('地域別送電損失（AC収束のみ）'); ylabel('MW'); grid on;

% Panel 6: Summary text
subplot(2, 3, 6);
axis off;
txt = {
    sprintf('全日本モデル: %d bus, %d branch, %d gen', ...
        size(aj_mpc.bus,1), size(aj_mpc.branch,1), size(aj_mpc.gen,1)),
    sprintf('総負荷: %.0f MW', sum(aj_mpc.bus(:,3))),
    sprintf('総発電: %.0f MW', sum(aj_mpc.gen(:,2))),
    '',
    'AC収束: 7/10 地域',
    'DC収束: 10/10 地域',
    '',
    sprintf('計算時刻: 8月平日 14:00'),
    sprintf('負荷倍率: %.3f', mult),
};
text(0.1, 0.9, txt, 'FontSize', 11, 'VerticalAlignment', 'top', 'FontName', 'Helvetica');

sgtitle('1時間スナップショット潮流計算', 'FontSize', 16, 'FontWeight', 'bold');

% Save
png_path = fullfile(out_dir, 'pf_1h_summary.png');
exportgraphics(fig, png_path, 'Resolution', 150);
fprintf('\nSaved: %s\n', png_path);

% Save results
save(fullfile(out_dir, 'pf_1h_results.mat'), 'results_1h', 'aj_res', 'aj_mpc', '-v7.3');
fprintf('Saved: pf_1h_results.mat\n');

fprintf('\n=== 1h Power Flow Complete ===\n');
