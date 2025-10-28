function multi_run_optimization
    % 粗加工参数设置
    lb = [1500, 300, 0.5];  % 下界（主轴转速, 进给量, 切削深度）
    ub = [3500, 900, 1];    % 上界
    weights_multi = [0.40, 0.3, 0.3];  % 多目标权重
    run_times = 10;         % 运行次数
    
    % 存储多目标优化结果
    best_params_multi = zeros(run_times, 3);  % 多目标参数
    best_obj_multi = zeros(run_times, 3);     % 多目标目标值
    
    % 存储单一目标优化结果（1:能耗, 2:铣削热, 3:切削力）
    best_params_single = cell(3, run_times);  % 单一目标参数（cell数组：{目标类型, 运行次数}）
    best_obj_single = cell(3, run_times);     % 单一目标目标值
    
    % 粗加工场景评估标准
    param_threshold = [8; 8; 8];        % 决策变量波动阈值(%)
    obj_threshold = [12; 12; 12];       % 目标函数COV阈值(%)
    
    % 1. 运行多目标优化
    fprintf('===== 开始多目标优化（能耗+铣削热+切削力）=====\n');
    for run = 1:run_times
        fprintf('\n===== 多目标运行第 %d 次 =====\n', run);
        [params, objectives] = rough_machining_optimize(lb, ub, weights_multi, 'multi');
        best_params_multi(run, :) = params;
        best_obj_multi(run, :) = objectives;
        fprintf('多目标第 %d 次运行完成\n', run);
    end
    
    % 2. 运行单一目标优化
    single_goals = {'能耗', '铣削热', '切削力'};  % 单一目标名称
    for goal_idx = 1:3
        fprintf('\n===== 开始单一目标优化（仅%s）=====\n', single_goals{goal_idx});
        for run = 1:run_times
            fprintf('\n===== 单一目标（%s）运行第 %d 次 =====\n', single_goals{goal_idx}, run);
            weights_single = zeros(1,3);
            weights_single(goal_idx) = 1;  % 仅当前目标权重为1，其他为0
            [params, objectives] = rough_machining_optimize(lb, ub, weights_single, 'single');
            best_params_single{goal_idx, run} = params;
            best_obj_single{goal_idx, run} = objectives;
            fprintf('单一目标（%s）第 %d 次运行完成\n', single_goals{goal_idx}, run);
        end
    end
    
    % 3. 整理单一目标结果为矩阵（便于计算和绘图）
    best_obj_single_mat = zeros(3, run_times, 3);  % [目标类型, 运行次数, 能耗/热/力]
    for goal_idx = 1:3
        for run = 1:run_times
            best_obj_single_mat(goal_idx, run, :) = best_obj_single{goal_idx, run};
        end
    end
    % 单一目标平均结果
    avg_obj_single = squeeze(mean(best_obj_single_mat, 2));  % 3×3（目标类型×指标）
    
    % 4. 多目标结果评估（同原逻辑）
    params_range = ub - lb;
    params_std_multi = std(best_params_multi);
    param_fluctuation_multi = params_std_multi ./ params_range * 100;
    
    obj_mean_multi = mean(best_obj_multi);
    obj_std_multi = std(best_obj_multi);
    obj_cov_multi = obj_std_multi ./ obj_mean_multi * 100;
    
    param_qualified_multi = param_fluctuation_multi <= param_threshold;
    obj_qualified_multi = obj_cov_multi <= obj_threshold;
    overall_qualified_multi = all(param_qualified_multi) & all(obj_qualified_multi);
    
    avg_params_multi = [
        round(mean(best_params_multi(:, 1))),
        round(mean(best_params_multi(:, 2))),
        mean(best_params_multi(:, 3))
    ];
    avg_obj_multi = round(mean(best_obj_multi, 1));
    
    % 5. 输出多目标评估结果
    fprintf('\n===== 多目标优化量化评估结果 =====\n');
    fprintf('粗加工场景标准：决策变量波动≤8%%，目标函数COV≤12%%\n\n');
 fprintf('决策变量波动评估：\n');
    for i = 1:3
        if params_std_multi(i)
            status = '合格';
        else
            status = '不合格';
        end
        fprintf('  %s: %.2f%% %s\n', get_param_name(i), param_fluctuation(i), status);
    end
    
    fprintf('\n目标函数波动评估：\n');
    for i = 1:3
        if obj_std_multi(i)
            status = '合格';
        else
            status = '不合格';
        end
        fprintf('  %s: %.2f%% %s\n', get_obj_name(i), obj_cov(i), status);
    end
    
    fprintf('\n整体评估: ');
    if overall_qualified
        fprintf('合格\n\n');
    else
        fprintf('不合格\n\n');
    end
    
    % 6. 输出多目标与单一目标平均结果对比
    fprintf('===== 多目标与单一目标平均结果对比 =====\n');
    fprintf('多目标平均参数: 主轴转速=%.0f r/min, 进给量=%.0f mm/min, 切削深度=%.2f mm\n', ...
        avg_params_multi(1), avg_params_multi(2), avg_params_multi(3));
    fprintf('多目标平均目标值: 能耗=%.0f J, 铣削热=%.0f ℃, 切削力=%.0f N\n', ...
        avg_obj_multi(1), avg_obj_multi(2), avg_obj_multi(3));
    
    for goal_idx = 1:3
        fprintf('%s单一优化平均目标值: 能耗=%.0f J, 铣削热=%.0f ℃, 切削力=%.0f N\n', ...
            single_goals{goal_idx}, ...
            round(avg_obj_single(goal_idx,1)), ...
            round(avg_obj_single(goal_idx,2)), ...
            round(avg_obj_single(goal_idx,3)));
    end
    
    % 7. 可视化整合：多目标+单一目标结果
    
    % 7.1 帕累托前沿图（多目标非支配解+单一目标最优解）
    % 获取多目标帕累托解（最后一次运行的结果，用于绘图）
    [~, ~, pareto_f_multi] = rough_machining_optimize(lb, ub, weights_multi, 'multi', true);
    % 单一目标最优解（取最后一次运行的结果）
    single_best_obj = zeros(3,3);
    for goal_idx = 1:3
        single_best_obj(goal_idx,:) = best_obj_single{goal_idx, run_times};
    end
    
    figure('Position', [200, 200, 900, 700]);
    % 多目标帕累托解
    scatter3(pareto_f_multi(:,1), pareto_f_multi(:,2), pareto_f_multi(:,3), 50, 'b', 'filled', 'DisplayName', '多目标非支配解');
    hold on;
    % 单一目标最优解（标记为星号）
    scatter3(single_best_obj(1,1), single_best_obj(1,2), single_best_obj(1,3), 100, 'r', 'p', 'LineWidth', 2, 'DisplayName', '仅优化能耗');
    scatter3(single_best_obj(2,1), single_best_obj(2,2), single_best_obj(2,3), 100, 'g', 'd', 'LineWidth', 2, 'DisplayName', '仅优化铣削热');
    scatter3(single_best_obj(3,1), single_best_obj(3,2), single_best_obj(3,3), 100, 'm', 's', 'LineWidth', 2, 'DisplayName', '仅优化切削力');
    xlabel('能耗 (J)'); ylabel('铣削热 (℃)'); zlabel('切削力 (N)');
    title('多目标与单一目标优化结果对比（帕累托前沿）');
    grid on; legend('Location', 'best'); view(30, 45);
    
    % 7.2 目标值趋势对比图（多目标+单一目标）
    figure('Position', [100, 100, 1200, 400]);
    % 能耗趋势
    subplot(1,3,1);
    plot(1:run_times, best_obj_multi(:,1), 'bo-', 'LineWidth', 2, 'DisplayName', '多目标');
    hold on;
    plot(1:run_times, squeeze(best_obj_single_mat(1,:,1)), 'ro-', 'LineWidth', 1.5, 'DisplayName', '仅能耗');
    plot(1:run_times, squeeze(best_obj_single_mat(2,:,1)), 'go-', 'LineWidth', 1.5, 'DisplayName', '仅铣削热');
    plot(1:run_times, squeeze(best_obj_single_mat(3,:,1)), 'mo-', 'LineWidth', 1.5, 'DisplayName', '仅切削力');
    xlabel('运行次数'); ylabel('能耗 (J)'); title('能耗优化趋势对比');
    grid on; legend('Location', 'best');
    
    % 铣削热趋势
    subplot(1,3,2);
    plot(1:run_times, best_obj_multi(:,2), 'bo-', 'LineWidth', 2, 'DisplayName', '多目标');
    hold on;
    plot(1:run_times, squeeze(best_obj_single_mat(1,:,2)), 'ro-', 'LineWidth', 1.5, 'DisplayName', '仅能耗');
    plot(1:run_times, squeeze(best_obj_single_mat(2,:,2)), 'go-', 'LineWidth', 1.5, 'DisplayName', '仅铣削热');
    plot(1:run_times, squeeze(best_obj_single_mat(3,:,2)), 'mo-', 'LineWidth', 1.5, 'DisplayName', '仅切削力');
    xlabel('运行次数'); ylabel('铣削热 (℃)'); title('铣削热优化趋势对比');
    grid on; legend('Location', 'best');
    
    % 切削力趋势
    subplot(1,3,3);
    plot(1:run_times, best_obj_multi(:,3), 'bo-', 'LineWidth', 2, 'DisplayName', '多目标');
    hold on;
    plot(1:run_times, squeeze(best_obj_single_mat(1,:,3)), 'ro-', 'LineWidth', 1.5, 'DisplayName', '仅能耗');
    plot(1:run_times, squeeze(best_obj_single_mat(2,:,3)), 'go-', 'LineWidth', 1.5, 'DisplayName', '仅铣削热');
    plot(1:run_times, squeeze(best_obj_single_mat(3,:,3)), 'mo-', 'LineWidth', 1.5, 'DisplayName', '仅切削力');
    xlabel('运行次数'); ylabel('切削力 (N)'); title('切削力优化趋势对比');
    grid on; legend('Location', 'best');
    
    % 7.3 平均目标值柱状对比图
    figure('Position', [100, 500, 1000, 500]);
    bar_groups = categorical({'能耗 (J)', '铣削热 (℃)', '切削力 (N)'});
    % 多目标平均
    multi_avg = avg_obj_multi;
    % 单一目标平均
    single_avg1 = avg_obj_single(1,:);  % 仅能耗
    single_avg2 = avg_obj_single(2,:);  % 仅铣削热
    single_avg3 = avg_obj_single(3,:);  % 仅切削力
    
    bar([multi_avg; single_avg1; single_avg2; single_avg3]', 0.7);
    set(gca, 'XTickLabel', bar_groups, 'FontSize', 10);
    legend('多目标', '仅优化能耗', '仅优化铣削热', '仅优化切削力', 'Location', 'best');
    title('多目标与单一目标平均优化结果对比');
    grid on; ylabel('平均目标值');
    
    % 7.4 参数趋势图（仅多目标，保持原逻辑）
    figure('Position', [100, 600, 1200, 400]);
    subplot(1,3,1);
    plot_qualified_trend(1:run_times, best_params_multi(:,1), param_fluctuation_multi(1), param_threshold(1), '主轴转速', 'r/min');
    subplot(1,3,2);
    plot_qualified_trend(1:run_times, best_params_multi(:,2), param_fluctuation_multi(2), param_threshold(2), '进给量', 'mm/min');
    subplot(1,3,3);
    plot_qualified_trend(1:run_times, best_params_multi(:,3), param_fluctuation_multi(3), param_threshold(3), '切削深度', 'mm');
end

% 优化函数（支持多目标/单一目标，返回帕累托解用于绘图）
function [best_param, best_obj, pareto_f] = rough_machining_optimize(lb, ub, weights, opt_type, return_pareto)
    % 参数默认值
    if nargin < 5, return_pareto = false; end  % 是否返回帕累托解
    max_gen = 10000;            % 最大迭代次数
    pop_size = 200;            % 种群大小
    
    % 1. 遗传算法初始化
    ga_options = optimoptions('gamultiobj', 'Display', 'off', ...
        'PopulationSize', pop_size, 'MaxGenerations', max_gen, ...
        'UseVectorized', true);
    
    % 2. 执行遗传算法获取初始Pareto解
    [ga_x, ga_f] = gamultiobj(@objective_functions, 3, [], [], [], [], lb, ub, ga_options);
    
    % 3. 黏菌算法优化（基于GA结果）
    sma_x = zeros(size(ga_x));
    sma_f = zeros(size(ga_f));
    for i = 1:size(ga_x, 1)
        sma_opts = struct('Max_iter', 3000, 'N', 100, 'lb', lb, 'ub', ub);
        [sma_x(i,:), sma_f(i,:)] = sma_optim(@objective_functions, ga_x(i,:), sma_opts);
    end
    
    % 4. 合并GA和SMA结果，提取帕累托解
    all_x = [ga_x; sma_x];
    all_f = [ga_f; sma_f];
    [pareto_x, pareto_f] = extract_pareto(all_x, all_f);
    
    % 5. 选择最优解（多目标用TOPSIS，单一目标直接取最小）
    if strcmp(opt_type, 'multi')
        % 多目标：TOPSIS评价
        [~, topsis_scores] = ew_topsis(pareto_f, weights);
        [~, best_idx] = max(topsis_scores);
    else
        % 单一目标：取目标值最小的解（权重最大的目标）
        [~, best_idx] = min(pareto_f(:, find(weights==1)));  % 单一目标权重为1
    end
    
    best_param = pareto_x(best_idx, :);
    best_obj = pareto_f(best_idx, :);
    
    % 若需要返回帕累托解（用于绘图）
    if ~return_pareto, pareto_f = []; end
end

% 以下函数与原代码一致（plot_qualified_trend、sma_optim、extract_pareto、objective_functions、ew_topsis、get_param_name、get_obj_name）
function plot_qualified_trend(x, y, actual, threshold, ylabel_text, unit)
    plot(x, y, 'o-', 'LineWidth', 2);
    title([ylabel_text '随运行次数变化 (波动: ' num2str(actual, '%.2f') '%']);
    xlabel('运行次数'); 
    ylabel([ylabel_text ' (' unit ')']);
    grid on;
    
    hold on;
    if actual <= threshold
        plot(x, ones(size(x)) * mean(y), 'g--', 'LineWidth', 1.5);
        legend('实际值', '合格阈值');
    else
        plot(x, ones(size(x)) * mean(y), 'r--', 'LineWidth', 1.5);
        legend('实际值', '不合格阈值');
    end
    
    for i = 1:length(x)
        plot(x(i), y(i), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    end
end

function [best_x, best_f] = sma_optim(obj_func, init_x, opts)
    Max_iter = opts.Max_iter;
    N = opts.N;
    dim = length(init_x);
    lb = opts.lb;
    ub = opts.ub;
    
    X = zeros(N, dim);
    for i = 1:N
        X(i,:) = lb + (ub - lb) .* rand(1, dim);
    end
    X(1,:) = init_x;
    
    fitness = zeros(N, 1);
    for i = 1:N
        f = obj_func(X(i,:));
        fitness(i) = sum(f.^2);
    end
    
    [min_fitness, min_idx] = min(fitness);
    best_x = X(min_idx,:);
    best_f = obj_func(best_x);
    
    for t = 1:Max_iter
        a = atanh(-(t/Max_iter)+1);
        b = 1 - t/Max_iter;
        z = 0.03;
        
        for i = 1:N
            if rand < z
                X(i,:) = lb + (ub - lb) .* rand(1, dim);
            else
                p = tanh(abs(fitness(i) - min_fitness));
                vb = unifrnd(-a, a, 1, dim);
                vc = unifrnd(-b, b, 1, dim);
                
                A = randi([1, N]);
                B = randi([1, N]);
                
                for j = 1:dim
                    r = rand();
                    if r < p
                        X(i,j) = best_x(j) + vb(j) * (X(A,j) - X(B,j));
                    else
                        X(i,j) = vc(j) * X(i,j);
                    end
                    X(i,j) = max(min(X(i,j), ub(j)), lb(j));
                end
            end
            
            new_f = obj_func(X(i,:));
            new_fitness = sum(new_f.^2);
            if new_fitness < fitness(i)
                fitness(i) = new_fitness;
                if fitness(i) < min_fitness
                    min_fitness = fitness(i);
                    best_x = X(i,:);
                    best_f = new_f;
                end
            end
        end
    end
end

function [pareto_x, pareto_f] = extract_pareto(all_x, all_f)
    if isempty(all_x) | isempty(all_f) | size(all_x, 1) ~= size(all_f, 1)
        fprintf('警告: 输入矩阵为空或行数不一致，返回空矩阵\n');
        pareto_x = [];
        pareto_f = [];
        return;
    end
    
    n = size(all_f, 1);
    is_pareto = true(n, 1);
    
    for i = 1:n
        for j = 1:n
            if i ~= j
                if all(all_f(j,:) <= all_f(i,:)) & any(all_f(j,:) < all_f(i,:))
                    is_pareto(i) = false;
                    break;
                end
            end
        end
    end
    
    is_pareto = logical(is_pareto);
    if any(is_pareto)
        pareto_x = all_x(is_pareto,:);
        pareto_f = all_f(is_pareto,:);
    else
        fprintf('警告: 未找到非支配解，返回第一个解\n');
        pareto_x = all_x(1,:);
        pareto_f = all_f(1,:);
    end
end

function f = objective_functions(x)
    [n, ~] = size(x);
    f = zeros(n, 3);
    
    for i = 1:n
        % 能耗模型
        f1 = 3.14053e5 + 15.14325*x(i,1) - 591.15282*x(i,2) - 23060.62366*x(i,3)...
           - 0.008820*x(i,1)*x(i,2) + 0.919341*x(i,1)*x(i,3) - 3.05767*x(i,2)*x(i,3)...
           - 0.001001*x(i,1)^2 + 0.345606*x(i,2)^2 + 19778.59248*x(i,3)^2;
        
        % 铣削热模型
        f2 = 15.88282 + 0.043578*x(i,1) + 0.190811*x(i,2) + 262.86326*x(i,3)...
           + 0.000017*x(i,1)*x(i,2) + 0.024828*x(i,1)*x(i,3) + 0.103949*x(i,2)*x(i,3)...
           - 6.32310e-06*x(i,1)^2 - 0.000116*x(i,2)^2 - 95.08290*x(i,3)^2;
        
        % 切削力模型
        f3 = -14.40625 - 0.403063*x(i,1) + 1.78646*x(i,2) + 1586.25000*x(i,3)...
           - 0.000527*x(i,1)*x(i,2) - 0.493500*x(i,1)*x(i,3) + 1.40833*x(i,2)*x(i,3)...
           + 0.000137*x(i,1)^2 - 0.000214*x(i,2)^2 - 106.00000*x(i,3)^2;
        
        f(i,:) = [f1, f2, f3];
    end
end

function [weights, topsis_scores] = ew_topsis(data, user_weights)
    data_positive = max(data) - data;
    normalized_data = data_positive ./ repmat(norm(data_positive), size(data_positive,1), 1);
    weights = user_weights;
    weighted_data = normalized_data .* repmat(weights, size(normalized_data,1), 1);
    ideal_best = max(weighted_data);
    ideal_worst = min(weighted_data);
    d_best = sqrt(sum((weighted_data - ideal_best).^2, 2));
    d_worst = sqrt(sum((weighted_data - ideal_worst).^2, 2));
    topsis_scores = d_worst ./ (d_best + d_worst + 1e-8);
end

function name = get_param_name(idx)
    switch idx
        case 1; name = '主轴转速';
        case 2; name = '进给量';
        case 3; name = '切削深度';
        otherwise; name = '未知参数';
    end
end

function name = get_obj_name(idx)
    switch idx
        case 1; name = '能耗';
        case 2; name = '铣削热';
        case 3; name = '切削力';
        otherwise; name = '未知目标';
    end
end