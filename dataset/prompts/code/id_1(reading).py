'''
2025年6月10日至6月12日，我想去苏州旅游，请帮我制定一个为期3天的旅游计划，我的预算是8500元，出发地点是北京西站。
6月10日上午从北京出发，6月12日晚从苏州乘坐高铁返回。此次行程希望安排高评分的景点、餐厅和住宿，市内交通以打车为主，整体行程要舒适、动线合理，
并能体验苏州的文化韵味。
'''

import math

import pyomo.environ as pyo
from datetime import datetime, timedelta
from pyomo.opt import TerminationCondition
import requests
from pyomo.opt import SolverStatus

# 用户输入参数
origin_city = "北京市"        # 出发城市
destination_city = "苏州市"   # 目的地城市
budget = 8500               # 总预算(元)
start_date = "2025年6月10日" # 出发日期
end_date = "2025年6月12日"   # 返回日期
travel_days = 3             # 旅游天数
peoples = 1                 # 旅游人数


# 获取旅游数据：从本地API服务器获取景点、酒店、餐厅、交通等数据
def fetch_data():
    url = "http://localhost:12457"
    # 获取城际交通数据(去程)
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    # 获取城际交通数据(返程)
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 获取目的地POI数据
    poi_data = {
        'attractions': requests.get(url + f"/attractions/{destination_city}").json(),      # 景点数据
        'accommodations': requests.get(url + f"/accommodations/{destination_city}").json(), # 住宿数据
        'restaurants': requests.get(url + f"/restaurants/{destination_city}").json()       # 餐厅数据
    }

    # 获取市内交通数据(不同POI之间的交通时间和成本)
    intra_city_trans = requests.get(url + f"/intra-city-transport/{destination_city}").json()
    return cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans


# 获取两点间交通参数的辅助函数
def get_trans_params(intra_city_trans, hotel_id, attr_id, param_type):
    """
    从交通数据中获取指定参数
    Args:
        intra_city_trans: 市内交通数据字典
        hotel_id: 酒店ID
        attr_id: 景点ID  
        param_type: 参数类型('taxi_duration', 'taxi_cost', 'bus_duration', 'bus_cost')
    Returns:
        对应的交通参数值
    """
    for key in [f"{hotel_id},{attr_id}", f"{attr_id},{hotel_id}"]:
        if key in intra_city_trans:
            data = intra_city_trans[key]
            return {
                'taxi_duration': float(data.get('taxi_duration')),  # 打车时长(分钟)
                'taxi_cost': float(data.get('taxi_cost')),          # 打车费用(元)
                'bus_duration': float(data.get('bus_duration')),    # 公交时长(分钟)
                'bus_cost': float(data.get('bus_cost'))             # 公交费用(元)
            }[param_type]


# 构建MILP旅游规划优化模型
def build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans):
    """
    构建混合整数线性规划模型来优化旅游计划
    目标：在预算和时间约束下，最大化旅游体验评分
    """
    model = pyo.ConcreteModel()

    # ==================== 定义集合 ====================
    days = list(range(1, travel_days + 1))  # 旅游天数集合 [1, 2, 3]
    model.days = pyo.Set(initialize=days)

    # 将列表数据转换为字典，便于后续索引
    attraction_dict = {a['id']: a for a in poi_data['attractions']}    # 景点字典
    hotel_dict = {h['id']: h for h in poi_data['accommodations']}      # 酒店字典
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}    # 餐厅字典
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}  # 去程火车字典
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}            # 返程火车字典

    # 定义模型中的集合
    model.attractions = pyo.Set(initialize=attraction_dict.keys())     # 景点ID集合
    model.accommodations = pyo.Set(initialize=hotel_dict.keys())       # 酒店ID集合
    model.restaurants = pyo.Set(initialize=restaurant_dict.keys())     # 餐厅ID集合
    model.train_departure = pyo.Set(initialize=train_departure_dict.keys())  # 去程火车班次集合
    model.train_back = pyo.Set(initialize=train_back_dict.keys())            # 返程火车班次集合

    # ==================== 定义参数 ====================
    # 景点参数：包含费用、评分、游览时长等信息
    model.attr_data = pyo.Param(
        model.attractions,
        initialize=lambda m, a: {
            'id': attraction_dict[a]['id'],           # 景点ID
            'name': attraction_dict[a]['name'],       # 景点名称
            'cost': float(attraction_dict[a]['cost']), # 门票费用(元)
            'type': attraction_dict[a]['type'],       # 景点类型
            'rating': float(attraction_dict[a]['rating']), # 评分(1-5分)
            'duration': float(attraction_dict[a]['duration']) # 游览时长(分钟)
        }
    )

    # 酒店参数：包含费用、评分等信息
    model.hotel_data = pyo.Param(
        model.accommodations,
        initialize=lambda m, h: {
            'id': hotel_dict[h]['id'],                # 酒店ID
            'name': hotel_dict[h]['name'],            # 酒店名称
            'cost': float(hotel_dict[h]['cost']),     # 每晚费用(元)
            'type': hotel_dict[h]['type'],            # 酒店类型
            'rating': float(hotel_dict[h]['rating']), # 评分(1-5分)
            'feature': hotel_dict[h]['feature']       # 酒店特色
        }
    )

    # 餐厅参数：包含费用、评分、用餐时长等信息
    model.rest_data = pyo.Param(
        model.restaurants,
        initialize=lambda m, r: {
            'id': restaurant_dict[r]['id'],           # 餐厅ID
            'name': restaurant_dict[r]['name'],       # 餐厅名称
            'cost': float(restaurant_dict[r]['cost']), # 人均费用(元)
            'type': restaurant_dict[r]['type'],       # 餐厅类型
            'rating': float(restaurant_dict[r]['rating']), # 评分(1-5分)
            'recommended_food': restaurant_dict[r]['recommended_food'], # 推荐菜品
            'queue_time': float(restaurant_dict[r]['queue_time']),      # 排队时间(分钟)
            'duration': float(restaurant_dict[r]['duration'])           # 用餐时长(分钟)
        }
    )

    # 去程火车参数：包含费用、时长、车站信息等
    model.train_departure_data = pyo.Param(
        model.train_departure,
        initialize=lambda m, t: {
            'train_number': train_departure_dict[t]['train_number'],         # 车次号
            'cost': float(train_departure_dict[t]['cost']),                  # 票价(元)
            'duration': float(train_departure_dict[t]['duration']),          # 行程时长(分钟)
            'origin_id': train_departure_dict[t]['origin_id'],               # 出发站ID
            'origin_station': train_departure_dict[t]['origin_station'],     # 出发站名
            'destination_id': train_departure_dict[t]['destination_id'],     # 到达站ID
            'destination_station': train_departure_dict[t]['destination_station'] # 到达站名
        }
    )

    # 返程火车参数：结构同去程
    model.train_back_data = pyo.Param(
        model.train_back,
        initialize=lambda m, t: {
            'train_number': train_back_dict[t]['train_number'],
            'cost': float(train_back_dict[t]['cost']),
            'duration': float(train_back_dict[t]['duration']),
            'origin_id': train_back_dict[t]['origin_id'],
            'origin_station': train_back_dict[t]['origin_station'],
            'destination_id': train_back_dict[t]['destination_id'],
            'destination_station': train_back_dict[t]['destination_station']
        }
    )

    # ==================== 定义决策变量 ====================
    # 二进制变量：表示每天是否选择某个景点 (1=选择, 0=不选择)
    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    
    # 二进制变量：表示是否选择某个酒店 (1=选择, 0=不选择)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    
    # 二进制变量：表示每天是否选择某个餐厅 (1=选择, 0=不选择)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    
    # 二进制变量：表示每天的交通方式 (0=公交, 1=打车)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)
    
    # 二进制变量：表示选择哪个去程火车班次 (1=选择, 0=不选择)
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    
    # 二进制变量：表示选择哪个返程火车班次 (1=选择, 0=不选择)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)

    # 辅助二进制变量：表示某天是否同时选择了某个景点和某个酒店
    # 用于计算景点-酒店间的交通成本和时间
    model.attr_hotel = pyo.Var(
        model.days, model.attractions, model.accommodations,
        domain=pyo.Binary,
        initialize=0,
        bounds=(0, 1)
    )

    # ==================== 约束条件 ====================
    
    # 辅助变量逻辑约束：确保attr_hotel[d,a,h]正确表示景点a和酒店h在第d天是否同时被选择
    # 约束1：如果景点a在第d天未被选择，则attr_hotel[d,a,h]=0
    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]

    # 约束2：如果酒店h未被选择，则attr_hotel[d,a,h]=0  
    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]

    # 约束3：如果景点a在第d天被选择且酒店h被选择，则attr_hotel[d,a,h]=1
    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

    # 将逻辑约束添加到模型中
    model.link_attr_hotel1 = pyo.Constraint(
        model.days, model.attractions, model.accommodations,
        rule=link_attr_hotel_rule1
    )
    model.link_attr_hotel2 = pyo.Constraint(
        model.days, model.attractions, model.accommodations,
        rule=link_attr_hotel_rule2
    )
    model.link_attr_hotel3 = pyo.Constraint(
        model.days, model.attractions, model.accommodations,
        rule=link_attr_hotel_rule3
    )

    # ==================== 目标函数 ====================
    def obj_rule(model):
        """
        目标函数：最大化(总评分 - 总成本)
        评分越高越好，成本越低越好
        """
        # 计算总评分：景点评分 + 餐厅评分 + 酒店评分
        rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating']
                     for d in model.days for a in model.attractions) + \
                 sum(model.select_rest[d, r] * model.rest_data[r]['rating']
                     for d in model.days for r in model.restaurants) + \
                 sum(model.select_hotel[h] * model.hotel_data[h]['rating']
                     for h in model.accommodations)
        
        # 计算各项成本
        # 酒店成本：每晚费用 × (总天数-1)，最后一天不住酒店
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                         for h in model.accommodations)
        
        # 景点门票成本
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                              for d in model.days for a in model.attractions)
        
        # 餐厅用餐成本
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                              for d in model.days for r in model.restaurants)
        # 市内交通成本：酒店到景点往返的交通费用
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                    # 打车成本：往返出租车费用
                    (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
            ) + \
                    # 公交成本：往返公交费用 × 人数
                    peoples * model.trans_mode[d] * (
                            get_trans_params(intra_city_trans, h, a, 'bus_cost') + \
                            get_trans_params(intra_city_trans, a, h, 'bus_cost')
                    )
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations)
        
        # 去程火车成本
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                   for t in model.train_departure)
        
        # 返程火车成本
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                              for t in model.train_back)
        
        # 目标函数 = 总评分 - 加权总成本
        # (peoples+1)//2: 酒店费用分摊系数，单人旅游时为1
        # peoples: 个人消费项目按人数计算
        return rating - transport_cost - (peoples+1) // 2 * hotel_cost - peoples * (
                      attraction_cost + restaurant_cost + train_departure_cost + train_back_cost)

    # 将目标函数添加到模型中，目标是最大化
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # ==================== 预算约束 ====================
    def budget_rule(model):
        """
        预算约束：总支出不能超过预算
        """
        # 计算各项支出(与目标函数中的成本计算相同)
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                         for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                              for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                              for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                    (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
            ) + \
                    peoples * model.trans_mode[d] * (
                            get_trans_params(intra_city_trans, h, a, 'bus_cost') + \
                            get_trans_params(intra_city_trans, a, h, 'bus_cost')
                    )
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations)
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                   for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                              for t in model.train_back)
        
        # 总支出 ≤ 预算
        return (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
                     attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    # 将预算约束添加到模型中
    model.budget_con = pyo.Constraint(rule=budget_rule)

    # ==================== 时间约束 ====================
    def time_rule(model, d):
        """
        每日时间约束：每天的活动时间+交通时间不能超过840分钟(14小时)
        """
        # 活动时间 = 景点游览时间 + 餐厅用餐时间(包括排队)
        activity_time = sum(
            model.select_attr[d, a] * model.attr_data[a]['duration']
            for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
            for r in model.restaurants
        )
        
        # 交通时间 = 酒店到景点往返的交通时间
        trans_time = sum(
            model.attr_hotel[d, a, h] * (
                    # 打车时间：往返出租车时长
                    (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_duration') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_duration')
            ) + \
                    # 公交时间：往返公交时长
                    model.trans_mode[d] * (
                            get_trans_params(intra_city_trans, h, a, 'bus_duration') + \
                            get_trans_params(intra_city_trans, a, h, 'bus_duration')
                    )
            )
            for a in model.attractions
            for h in model.accommodations
        )
        
        # 总时间 ≤ 840分钟
        return activity_time + trans_time <= 840

    # 将时间约束添加到模型中
    model.time_con = pyo.Constraint(model.days, rule=time_rule)

    # ==================== 选择约束 ====================
    
    # 每个景点最多只能游览一次
    model.unique_attr = pyo.Constraint(
        model.attractions,
        rule=lambda m, a: sum(m.select_attr[d, a] for d in m.days) <= 1
    )

    # 每个餐厅最多只能去一次
    model.unique_rest = pyo.Constraint(
        model.restaurants,
        rule=lambda m, r: sum(m.select_rest[d, r] for d in m.days) <= 1
    )

    # 每天必须选择1个景点
    model.min_attr = pyo.Constraint(
        model.days,
        rule=lambda m, d: sum(m.select_attr[d, a] for a in m.attractions) == 1
    )

    # 每天必须选择3个餐厅(早中晚餐)
    model.min_rest = pyo.Constraint(
        model.days,
        rule=lambda m, d: sum(m.select_rest[d, r] for r in m.restaurants) == 3
    )

    # 必须选择1个去程火车班次
    model.single_train_departure = pyo.Constraint(
        rule=lambda m: sum(m.select_train_departure[t] for t in m.train_departure) == 1
    )

    # 必须选择1个返程火车班次
    model.single_train_back = pyo.Constraint(
        rule=lambda m: sum(m.select_train_back[t] for t in m.train_back) == 1
    )

    # 必须选择1个酒店
    model.single_hotel = pyo.Constraint(
        rule=lambda m: sum(m.select_hotel[h] for h in m.accommodations) == 1
    )

    # 交通方式约束：根据题目要求，市内交通以打车为主(trans_mode[d]=0表示打车)
    # 这里设置为1可能是代码错误，应该是0，但保持原代码逻辑
    model.transport_type = pyo.Constraint(
        model.days,
        rule=lambda m, d: m.trans_mode[d] == 1  # 市内交通以打车为主
    )

    return model


# ==================== 结果处理模块 ====================
# 以下函数用于从MILP求解结果中提取和处理旅游计划

# 生成旅游日期范围
def generate_date_range(start_date, end_date, date_format="%Y年%m月%d日"):
    """
    根据开始和结束日期生成旅游期间的日期列表
    Args:
        start_date: 开始日期字符串
        end_date: 结束日期字符串  
        date_format: 日期格式
    Returns:
        日期字符串列表，如['2025年6月10日', '2025年6月11日', '2025年6月12日']
    """
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    days = travel_days
    return [
        (start + timedelta(days=i)).strftime(date_format)
        for i in range(days)
    ]


# 从求解结果中提取选择的火车班次
def get_selected_train(model, train_type='departure'):
    """
    从MILP求解结果中提取被选中的火车班次
    Args:
        model: Pyomo求解后的模型
        train_type: 'departure'(去程) 或 'back'(返程)
    Returns:
        选中的火车班次数据字典
    """
    if train_type not in ['departure', 'back']:
        raise ValueError("train_type must in ['departure', 'back']")

    # 根据类型选择对应的数据集和变量
    train_set = model.train_departure if train_type == 'departure' else model.train_back
    train_data = model.train_departure_data if train_type == 'departure' else model.train_back_data
    
    # 查找变量值>0.9的火车班次(由于数值误差，用0.9而不是1)
    selected_train = [
        train_data[t]
        for t in train_set
        if pyo.value(
            model.select_train_departure[t] if train_type == 'departure'
            else model.select_train_back[t]
        ) > 0.9
    ]
    return selected_train[0]  # 返回唯一选中的班次


# 从求解结果中提取选择的POI(景点或餐厅)
def get_selected_poi(model, type, day, selected_poi, k=1):
    """
    从MILP求解结果中提取某天被选中的POI
    Args:
        model: Pyomo求解后的模型
        type: 'restaurant'(餐厅) 或 'attraction'(景点)
        day: 指定天数
        selected_poi: 已选择的POI ID列表(用于避重)
        k: 选择数量(未使用)
    Returns:
        选中的POI数据列表
    """
    # 根据类型选择对应的数据集和变量
    if type == 'restaurant':
        poi_set = model.restaurants
        poi_data = model.rest_data
        select_set = model.select_rest
    else:
        poi_set = model.attractions
        poi_data = model.attr_data
        select_set = model.select_attr

    # 查找该天被选中且未重复的POI
    selected_poi = [
        poi_data[t]
        for t in poi_set
        if t not in selected_poi and pyo.value(select_set[day, t]) > 0.9
    ]
    return selected_poi


# 从求解结果中提取选择的酒店
def get_selected_hotel(model):
    """
    从MILP求解结果中提取被选中的酒店
    Args:
        model: Pyomo求解后的模型
    Returns:
        选中的酒店数据字典
    """
    selected_hotel = [
        model.hotel_data[t]
        for t in model.accommodations
        if pyo.value(model.select_hotel[t]) > 0.9  # 查找变量值>0.9的酒店
    ]
    return selected_hotel[0]  # 返回唯一选中的酒店


# 计算每日活动的总时间
def get_time(model, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day, intra_city_trans):
    """
    计算某天的总活动时间(活动时间 + 交通时间)
    Args:
        model: Pyomo模型
        selected_attr: 选中的景点信息
        selected_rest: 选中的餐厅列表
        departure_trains: 去程火车信息
        back_trains: 返程火车信息  
        selected_hotel: 选中的酒店信息
        day: 指定天数
        intra_city_trans: 市内交通数据
    Returns:
        tuple: (总时间, 交通时间)
    """
    daily_time = 0
    
    # 累加景点游览时间
    daily_time += selected_attr['duration']
    
    # 累加餐厅用餐时间(包括排队时间)
    for r in selected_rest:
        daily_time += r['queue_time'] + r['duration']

    # 根据交通方式计算往返交通时间
    if pyo.value(model.trans_mode[day]) > 0.9:  # 选择公交
        transport_time = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'bus_duration'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'bus_duration'
        )
    else:  # 选择打车
        transport_time = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'taxi_duration'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'taxi_duration'
        )

    return daily_time + transport_time, transport_time


# 计算每日活动的总费用
def get_cost(model, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day, intra_city_trans):
    """
    计算某天的总费用(门票 + 用餐 + 交通 + 住宿 + 火车票)
    Args:
        model: Pyomo模型
        selected_attr: 选中的景点信息
        selected_rest: 选中的餐厅列表
        departure_trains: 去程火车信息
        back_trains: 返程火车信息
        selected_hotel: 选中的酒店信息
        day: 指定天数
        intra_city_trans: 市内交通数据
    Returns:
        tuple: (总费用, 交通费用)
    """
    daily_cost = 0
    
    # 景点门票费用(按人数计算)
    daily_cost += peoples * selected_attr['cost']
    
    # 餐厅用餐费用(按人数计算)
    for r in selected_rest:
        daily_cost += peoples * r['cost']

    # 根据交通方式计算往返交通费用
    if pyo.value(model.trans_mode[day]) > 0.9:  # 选择公交
        transport_cost = peoples * get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'bus_cost'
        ) + peoples * get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'bus_cost'
        )
    else:  # 选择打车
        transport_cost = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'taxi_cost'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'taxi_cost'
        )

    # 住宿费用(最后一天不住酒店)
    if day != travel_days:
        daily_cost += selected_hotel['cost']
        
    # 火车票费用(第一天加去程，最后一天加返程)
    if day == 1:
        daily_cost += peoples * departure_trains['cost']
    if day == travel_days:
        daily_cost += peoples * back_trains['cost']
        
    return daily_cost + transport_cost, transport_cost

# 生成简化的POI选择结果(用于特定输出格式)
def generate_poi(model, intra_city_trans):
    """
    生成简化版的旅游计划结果，主要包含选中的POI ID列表
    Args:
        model: Pyomo求解后的模型
        intra_city_trans: 市内交通数据
    Returns:
        包含查询信息和选中POI的字典
    """
    # 提取选中的火车班次和酒店
    departure_trains = get_selected_train(model, 'departure')
    back_trains = get_selected_train(model, 'back')
    selected_hotel = get_selected_hotel(model)['id']

    # 初始化选择列表
    select_at = []  # 景点ID列表
    select_re = []  # 餐厅ID列表
    
    # 遍历每一天，提取选中的景点和餐厅
    for day in sorted(model.days):
        attr_details = []
        rest_details = []
        
        # 获取该天选中的景点(每天1个)
        attr_details = get_selected_poi(model, 'attraction', day, select_at)[0]
        select_at.append(attr_details['id'])
        
        # 获取该天选中的餐厅(每天3个)
        rest_details = get_selected_poi(model, 'restaurant', day, select_re)
        for r in rest_details:
            select_re.append(r['id'])

    # 返回结构化的查询结果
    return {
        "query_id": "1",
        "query": "2025年6月10日至6月12日，我想去苏州旅游，请帮我制定一个为期3天的旅游计划，我的预算是8500元，出发地点是北京西站。6月10日上午从北京出发，6月12日晚从苏州乘坐高铁返回。此次行程希望安排高评分的景点、餐厅和住宿，市内交通以打车为主，整体行程要舒适、动线合理，并能体验苏州的文化韵味。",
        "travel_days":travel_days,
        "budget": budget,
        "peoples": peoples,
        "origin_city": origin_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "departure_trains": {
            'train_number': departure_trains['train_number'],
            'origin_id': departure_trains['origin_id'],
            'destination_id': departure_trains['destination_id']
        },
        "back_trains": {
            'train_number': back_trains['train_number'],
            'origin_id': back_trains['origin_id'],
            'destination_id': back_trains['destination_id']
        },
        "hotels": selected_hotel,
        "attractions": select_at,
        "restaurants": select_re
    }

# ==================== 主要结果生成函数 ====================
# 生成详细的每日旅游计划
def generate_daily_plan(model, intra_city_trans):
    """
    生成完整的每日旅游计划，包含详细的时间、费用、POI信息
    这是结果处理的核心函数，调用其他辅助函数来构建完整的旅游计划
    
    Args:
        model: Pyomo求解后的模型，包含所有决策变量的最优解
        intra_city_trans: 市内交通数据字典
    Returns:
        完整的旅游计划字典，包含每日详细安排
    """
    # 1. 提取全局选择结果
    departure_trains = get_selected_train(model, 'departure')  # 去程火车
    back_trains = get_selected_train(model, 'back')           # 返程火车  
    selected_hotel = get_selected_hotel(model)                # 选中的酒店
    
    # 2. 初始化计划数据
    total_cost = 0      # 总费用累计
    daily_plans = []    # 每日计划列表
    select_at = []      # 已选景点ID列表(避免重复)
    select_re = []      # 已选餐厅ID列表(避免重复)
    date = generate_date_range(start_date, end_date)  # 生成日期列表
    
    # 3. 遍历每一天，生成详细的每日计划
    for day in sorted(model.days):
        # 3.1 提取该天选中的POI
        attr_details = get_selected_poi(model, 'attraction', day, select_at)[0]  # 该天的景点
        select_at.append(attr_details['id'])  # 记录已选景点
        
        rest_details = get_selected_poi(model, 'restaurant', day, select_re)     # 该天的餐厅(3个)
        for r in rest_details:
            select_re.append(r['id'])  # 记录已选餐厅
            
        # 3.2 分配三餐(假设按顺序分配为早中晚餐)
        meal_allocation = {
            'breakfast': rest_details[0],  # 早餐
            'lunch': rest_details[1],      # 午餐
            'dinner': rest_details[2]      # 晚餐
        }

        # 3.3 计算该天的时间和费用
        daily_time, transport_time = get_time(model, attr_details, rest_details, departure_trains, back_trains,
                                              selected_hotel, day, intra_city_trans)
        daily_cost, transport_cost = get_cost(model, attr_details, rest_details, departure_trains, back_trains,
                                              selected_hotel, day, intra_city_trans)
        
        # 3.4 构建该天的计划数据结构
        day_plan = {
            "date": f"{date[day - 1]}",                    # 日期
            "cost": round(daily_cost, 2),                  # 当日总费用
            "cost_time": round(daily_time, 2),             # 当日总时间(分钟)
            "hotel": selected_hotel if day != travel_days else "null",  # 住宿(最后一天不住)
            "attractions": attr_details,                   # 景点详情
            "restaurants": [                               # 餐厅详情(按三餐分类)
                {
                    "type": meal_type,
                    "restaurant": rest if rest else None
                } for meal_type, rest in meal_allocation.items()
            ],
            "transport": {                                 # 交通信息
                "mode": "bus" if pyo.value(model.trans_mode[day]) > 0.9 else "taxi",  # 交通方式
                "cost": round(transport_cost, 2),          # 交通费用
                "duration": round(transport_time, 2)       # 交通时间
            }
        }
        daily_plans.append(day_plan)
        total_cost += daily_cost  # 累计总费用

    # 4. 构建并返回完整的旅游计划
    return {
        "budget": budget,                              # 预算
        "peoples": peoples,                            # 人数
        "travel_days": travel_days,                    # 旅游天数
        "origin_city": origin_city,                    # 出发城市
        "destination_city": destination_city,          # 目的地城市
        "start_date": start_date,                      # 开始日期
        "end_date": end_date,                          # 结束日期
        "daily_plans": daily_plans,                    # 每日详细计划
        "departure_trains": departure_trains,          # 去程火车信息
        "back_trains": back_trains,                    # 返程火车信息
        "total_cost": round(total_cost, 2),           # 总费用
        "objective_value": round(pyo.value(model.obj), 2)  # 目标函数值
    }


# ==================== 求解器配置和主执行模块 ====================

# 配置SCIP求解器
def configure_solver():
    """
    配置SCIP混合整数线性规划求解器
    设置求解参数以获得高质量的解
    """
    solver = pyo.SolverFactory('scip')
    solver.options = {
        'limits/time': 300,  # 求解时间限制：300秒
        'limits/gap': 0,     # 最优性间隙：0表示寻找最优解
    }
    return solver


# 主求解函数：求解旅行规划问题
def solve_travel_plan(data):
    """
    旅游规划优化问题的主执行函数
    整合模型构建、求解和结果处理的完整流程
    
    Args:
        data: 元组，包含(去程火车数据, 返程火车数据, POI数据, 市内交通数据)
    
    执行流程:
        1. 构建MILP优化模型
        2. 配置和运行SCIP求解器  
        3. 提取和格式化求解结果
        4. 输出结构化的旅游计划
    """
    # 1. 解包输入数据
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = data
    
    # 2. 构建MILP优化模型
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    # 3. 配置求解器并求解
    solver = configure_solver()
    results = solver.solve(model, tee=True)  # tee=True显示求解过程
    
    # 4. 生成并输出旅游计划
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")


if __name__ == "__main__":
    data = fetch_data()
    solve_travel_plan(data)
