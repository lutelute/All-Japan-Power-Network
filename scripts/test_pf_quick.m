%% Quick PF convergence test for all regions + All-Japan
% Run: matlab -batch "run('scripts/test_pf_quick.m')"

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

mat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'output', 'matpower_alljapan');
regions = {'hokkaido','tohoku','tokyo','chubu','hokuriku', ...
           'kansai','chugoku','shikoku','kyushu','okinawa','alljapan'};

fprintf('\n%-12s %6s %6s %6s  %10s  %s\n', 'Region', 'Bus', 'Gen', 'Branch', 'Pd(MW)', 'AC/DC');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:length(regions)
    r = regions{i};
    mat_file = fullfile(mat_dir, [r '.mat']);
    if ~isfile(mat_file)
        fprintf('%-12s  MISSING\n', r);
        continue
    end
    mpc = loadcase(mat_file);
    nb = size(mpc.bus, 1);
    ng = size(mpc.gen, 1);
    nbr = size(mpc.branch, 1);
    pd = sum(mpc.bus(:, 3));

    % Try AC first
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.nr.max_it', 20);
    result = runpf(mpc, mpopt);
    if result.success
        mode = 'AC';
    else
        mpopt_dc = mpoption(mpopt, 'model', 'DC');
        result = rundcpf(mpc, mpopt_dc);
        if result.success
            mode = 'DC';
        else
            mode = 'FAIL';
        end
    end
    fprintf('%-12s %6d %6d %6d  %10.0f  %s\n', r, nb, ng, nbr, pd, mode);
end
fprintf('\nDone.\n');
