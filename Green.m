% 输入参数设置
params.n = 2949;        % 主轴转速(r/min)
params.vf = 300;       % 进给量(mm/min)
params.ap = 0.5;         % 切削深度(mm)
params.energy = 190269; % 能耗(J)
params.heat = 269;      % 铣削热(℃)
params.force = 289;     % 切削力(N)
params.material = '45_steel'; % 材料
params.ae = 9;              % 切削宽度(mm)
params.L = 900;              % 切削路径长度(mm)

% 计算绿色指标
[carbon_emission, green_index] = zbjs(params); 