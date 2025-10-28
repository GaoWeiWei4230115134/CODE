function [carbon_emission, green_index] = zbjs(params)
    % 绿色制造指标计算函数
    % 输入参数: params 结构体，包含以下字段:
    %   n: 主轴转速(r/min), vf: 进给量(mm/min), ap: 切削深度(mm)
    %   energy: 能耗(J), heat: 铣削热(℃), force: 切削力(N)
    %   material: 材料类型('45_steel')，默认值为'45_steel'
    %   ae: 切削宽度(mm)，默认值为10
    %   L: 切削路径长度(mm)，默认值为100

    % 参数默认值设置
    if nargin < 1
        error('请输入包含切削参数的结构体');
    end
    
    % 提取参数，设置默认值
    n = params.n;
    vf = params.vf;
    ap = params.ap;
    energy = params.energy;
    heat = params.heat;
    force = params.force;
    material = params.material;
    ae = params.ae;
    L = params.L;
    
    % 处理未输入的参数
    if ~isfield(params, 'ae'), ae = 10; end
    if ~isfield(params, 'L'), L = 100; end
    if ~isfield(params, 'material'), material = '45_steel'; end
    
    % 1. 计算加工时间(min)
    machining_time = calculate_machining_time(vf, ap, ae, L);
    
    % 2. 能耗单位转换: J -> kWh
    energy_kWh = energy / (3.6e6);  % 1 kWh = 3.6e6 J
    
    % 3. 计算碳排放(kgCO2)
    carbon_emission = calculate_carbon_emission(energy_kWh);
    
    % 4. 计算绿色综合指标
    green_index = calculate_green_index(energy, carbon_emission, heat, force, material);
    
    % 5. 结果输出
    display_results(n, vf, ap, energy, heat, force, carbon_emission, green_index, machining_time);
    
    % 6. 绘制指标雷达图（完全兼容旧版本MATLAB）
    plot_indicator_radar(energy, carbon_emission, heat, force, material);
end

function t = calculate_machining_time(vf, ap, ae, L)
    % 计算加工时间
    t = (L * ae) / (vf * ap);
end

function ce = calculate_carbon_emission(energy_kWh)
    % 计算碳排放量
    % 中国电网碳排放因子(2023年平均): 0.58 kgCO2/kWh
    carbon_factor = 0.58;  % kgCO2/kWh
    ce = energy_kWh * carbon_factor;
end

function gi = calculate_green_index(energy, ce, heat, force, material)
    % 计算绿色综合指标
    
    % 指标理想值和阈值(可根据工艺要求调整)
    switch lower(material)
        case '45_steel'
            % 45号钢加工指标阈值
            E0 = 62785;       % 能耗理想值(J)
            Emax = 200000;     % 能耗阈值(J)
            Ce0 = 0.010115;        % 碳排放理想值(kgCO2)
            Cemax = 0.1;       % 碳排放阈值(kgCO2)
            T0 = 263;          % 铣削热理想值(℃)
            Tmax = 600;        % 铣削热阈值(℃)
            F0 = 289;          % 切削力理想值(N)
            Fmax = 2000;       % 切削力阈值(N)
        otherwise
            % 默认值
            E0 = 150000;
            Emax = 350000;
            Ce0 = 0.03;
            Cemax = 0.12;
            T0 = 220;
            Tmax = 320;
            F0 = 550;
            Fmax = 850;
    end
    
    % 指标归一化(极小型指标: 值越小越好)
    I_E = normalize_index(energy, E0, Emax);
    I_Ce = normalize_index(ce, Ce0, Cemax);
    I_T = normalize_index(heat, T0, Tmax);
    I_F = normalize_index(force, F0, Fmax);
    
    % 权重设置(可根据评估重点调整)
    w = [0.3, 0.3, 0.2, 0.2];  % 能耗, 碳排放, 铣削热, 切削力
    
    % 计算绿色综合指标(0-1)
    gi = w(1)*I_E + w(2)*I_Ce + w(3)*I_T + w(4)*I_F;
end

function norm_val = normalize_index(x, x0, xmax)
    % 极小型指标归一化(0-1)
    if x <= x0
        norm_val = 1;
    elseif x >= xmax
        norm_val = 0;
    else
        norm_val = 1 - (x - x0) / (xmax - x0);
    end
end

function display_results(n, vf, ap, energy, heat, force, ce, gi, t)
    % 结果显示
    fprintf('\n===== 45号钢切削加工绿色制造指标计算 =====\n');
    fprintf('切削参数:\n');
    fprintf('主轴转速 = %.4f r/min\n', n);
    fprintf('进给量 = %.4f mm/min\n', vf);
    fprintf('切削深度 = %.4f mm\n', ap);
    fprintf('加工时间 = %.4f min\n', t);
    fprintf('\n性能指标:\n');
    fprintf('能耗 = %.4f J\n', energy);
    fprintf('铣削热 = %.4f ℃\n', heat);
    fprintf('切削力 = %.4f N\n', force);
    fprintf('\n绿色制造指标:\n');
    fprintf('碳排放量 = %.6f kgCO2\n', ce);
    fprintf('绿色综合指标 = %.4f (0-1, 越高越好)\n', gi);
    
    % 指标等级评估
    if gi >= 0.8
        fprintf('指标等级: A级 (优秀)\n');
    elseif gi >= 0.7
        fprintf('指标等级: B级 (良好)\n');
    elseif gi >= 0.6
        fprintf('指标等级: C级 (中等)\n');
    else
        fprintf('指标等级: D级 (需改进)\n');
    end
end

function plot_indicator_radar(energy, ce, heat, force, material)
    % 绘制指标雷达图（完全兼容旧版本MATLAB）
    
    % 创建图形窗口
    fig = figure('Name', '绿色制造指标雷达图', 'Position', [100, 100, 600, 600]);
    
    % 检查MATLAB版本，选择合适的绘图方式
    ver = verLessThan('matlab', 'R2016b');
    
    % 指标名称
    indicators = {'', '', '', ''};
    
    % 计算各指标归一化值(0-1)
    switch lower(material)
        case '45_steel'
            E0 = 70000; Emax = 200000;
            Ce0 = 0.02; Cemax = 0.1;
            T0 = 300; Tmax = 600;
            F0 = 500; Fmax = 2000;
    end
    
    I_E = normalize_index(energy, E0, Emax);
    I_Ce = normalize_index(ce, Ce0, Cemax);
    I_T = normalize_index(heat, T0, Tmax);
    I_F = normalize_index(force, F0, Fmax);
    values = [I_E,I_Ce, I_T, I_F];
    
    % 雷达图数据准备
    n = length(indicators);
    theta = linspace(0, 2*pi, n);
    values = [values, values(1)];  % 手动闭合数据
    theta = [theta, theta(1)];     % 手动闭合角度
    
    if ver
        % 旧版本MATLAB（R2016b之前）使用polar函数
        polarplot(theta, values, 'LineWidth', 4);
        
        % 手动绘制径向网格线
        for r = 0.2:0.2:1
            theta_grid = linspace(0, 2*pi, 100);
            r_grid = r * ones(size(theta_grid));
            hold on;
            polarplot(theta_grid, r_grid, 'k--', 'LineWidth', 1);
        end
        
        % 手动绘制周向网格线
        for t = 0:pi/4:2*pi
            r_grid = linspace(0, 1, 100);
            theta_grid = t * ones(size(r_grid));
            hold on;
            polarplot(theta_grid, r_grid, 'k--', 'LineWidth', 0.5);
        end
        
        % 设置极坐标范围
        axis([0, 2*pi, 0, 1.1]);
        
    else
        % 新版本MATLAB使用polarplot函数
        ax = polaraxes(fig);
        polarplot(ax, theta, values, 'LineWidth', 2);
        
        % 手动绘制径向网格线
        for r = 0.2:0.2:1
            theta_grid = linspace(0, 2*pi, 100);
            r_grid = r * ones(size(theta_grid));
            hold(ax, 'on');
            polarplot(ax, theta_grid, r_grid, 'k--', 'LineWidth', 1);
        end
        
        % 手动绘制周向网格线
        for t = 0:pi/4:2*pi
            r_grid = linspace(0, 1, 100);
            theta_grid = t * ones(size(r_grid));
            hold(ax, 'on');
            polarplot(ax, theta_grid, r_grid, 'k--', 'LineWidth', 0.5);
        end
        
        % 设置极坐标范围
        rlim(ax, [0, 1.1]);
    end
    
    % 添加指标名称标签
    for i = 1:n
        % 计算标签位置（半径略大于1）
        label_r = 1.05;
        label_x = label_r * cos(theta(i));
        label_y = label_r * sin(theta(i));
        
        if ver
            % 旧版本使用text函数
            text(label_x, label_y, indicators{i}, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
        else
            % 新版本使用text函数并指定坐标系
            text(ax, label_x, label_y, indicators{i}, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
        end
    end
    
    % 标注数值
    for i = 1:n
        if ver
            text(theta(i), values(i)+0.05, num2str(round(values(i)*100)/100), ...
                'HorizontalAlignment', 'center', 'FontSize',10.5);
        else
            text(ax, theta(i), values(i)+0.05, num2str(round(values(i)*100)/100), ...
                'HorizontalAlignment', 'center', 'FontSize', 10.5);
        end
    end
    
    % 添加标题
    if ver
        title('绿色制造指标雷达图', 'FontSize', 12);
    else
        title(ax, '绿色制造指标雷达图', 'FontSize', 12);
    end
    
    hold off;
end