import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "深圳市"
destination_city = "上海市"
budget = 7000
start_date = "2025年06月10日"
end_date = "2025年06月12日"
travel_days = 3
peoples = 2

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()
    print(cross_city_train_back)
    # 获取POI数据并筛选符合要求的
    attractions = [a for a in requests.get(url + f"/attractions/{destination_city}").json() 
                  if float(a['rating']) >= 4.5 and float(a['cost']) <= 800]
    accommodations = [h for h in requests.get(url + f"/accommodations/{destination_city}").json() 
                     if float(h['rating']) >= 4.5 and float(h['cost']) <= 800 and '连锁' in h['type']]
    restaurants = requests.get(url + f"/restaurants/{destination_city}").json()

    poi_data = {
        'attractions': attractions,
        'accommodations': accommodations,
        'restaurants': restaurants
    }

    intra_city_trans = requests.get(url + f"/intra-city-transport/{destination_city}").json()
    return cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans

def get_trans_params(intra_city_trans, hotel_id, attr_id, param_type):
    for key in [f"{hotel_id},{attr_id}", f"{attr_id},{hotel_id}"]:
        if key in intra_city_trans:
            data = intra_city_trans[key]
            return {
                'taxi_duration': float(data.get('taxi_duration')),
                'taxi_cost': float(data.get('taxi_cost')),
                'bus_duration': float(data.get('bus_duration')),
                'bus_cost': float(data.get('bus_cost'))
            }[param_type]

def build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans):
    model = pyo.ConcreteModel()

    # 定义集合
    days = list(range(1, travel_days + 1))
    model.days = pyo.Set(initialize=days)

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations']}
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}

    model.attractions = pyo.Set(initialize=attraction_dict.keys())
    model.accommodations = pyo.Set(initialize=hotel_dict.keys())
    model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
    model.train_departure = pyo.Set(initialize=train_departure_dict.keys())
    model.train_back = pyo.Set(initialize=train_back_dict.keys())

    # 定义参数
    model.attr_data = pyo.Param(
        model.attractions,
        initialize=lambda m, a: {
            'id': attraction_dict[a]['id'],
            'name': attraction_dict[a]['name'],
            'cost': float(attraction_dict[a]['cost']),
            'type': attraction_dict[a]['type'],
            'rating': float(attraction_dict[a]['rating']),
            'duration': float(attraction_dict[a]['duration'])
        }
    )

    model.hotel_data = pyo.Param(
        model.accommodations,
        initialize=lambda m, h: {
            'id': hotel_dict[h]['id'],
            'name': hotel_dict[h]['name'],
            'cost': float(hotel_dict[h]['cost']),
            'type': hotel_dict[h]['type'],
            'rating': float(hotel_dict[h]['rating']),
            'feature': hotel_dict[h]['feature']
        }
    )

    model.rest_data = pyo.Param(
        model.restaurants,
        initialize=lambda m, r: {
            'id': restaurant_dict[r]['id'],
            'name': restaurant_dict[r]['name'],
            'cost': float(restaurant_dict[r]['cost']),
            'type': restaurant_dict[r]['type'],
            'rating': float(restaurant_dict[r]['rating']),
            'recommended_food': restaurant_dict[r]['recommended_food'],
            'queue_time': float(restaurant_dict[r]['queue_time']),
            'duration': float(restaurant_dict[r]['duration'])
        }
    )

    model.train_departure_data = pyo.Param(
        model.train_departure,
        initialize=lambda m, t: {
            'train_number': train_departure_dict[t]['train_number'],
            'cost': float(train_departure_dict[t]['cost']),
            'duration': float(train_departure_dict[t]['duration']),
            'origin_id': train_departure_dict[t]['origin_id'],
            'origin_station': train_departure_dict[t]['origin_station'],
            'destination_id': train_departure_dict[t]['destination_id'],
            'destination_station': train_departure_dict[t]['destination_station']
        }
    )
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

    # 定义变量
    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 1 for bus, 0 for taxi
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)

    model.attr_hotel = pyo.Var(
        model.days, model.attractions, model.accommodations,
        domain=pyo.Binary,
        initialize=0,
        bounds=(0, 1)
    )

    # 约束条件：景点与酒店的选择关系
    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]

    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]

    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

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

    # 每日选择一个景点
    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attr_per_day = pyo.Constraint(model.days, rule=one_attr_per_day_rule)

    # 景点不重复
    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 每日选择三个餐厅
    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_rest_per_day = pyo.Constraint(model.days, rule=three_rest_per_day_rule)

    # 餐厅不重复
    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_rest = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 选择一个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 选择一趟去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    # 选择一趟返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                       for r in model.restaurants)
        trans_time = sum(
            model.attr_hotel[d, a, h] * (
                (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_duration') + 
                    get_trans_params(intra_city_trans, a, h, 'taxi_duration')
                ) + 
                model.trans_mode[d] * (
                    get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                    get_trans_params(intra_city_trans, a, h, 'bus_duration')
                )
            )
            for a in model.attractions
            for h in model.accommodations
        )
        return attr_time + rest_time + trans_time <= 840

    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    # 预算约束
    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                         for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                              for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                              for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + 
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
                ) + 
                peoples * model.trans_mode[d] * (
                    get_trans_params(intra_city_trans, h, a, 'bus_cost') + 
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
        return (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
                     attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    model.budget = pyo.Constraint(rule=budget_rule)

    # 目标函数：最大化评分
    def obj_rule(model):
        attr_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating']
                         for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating']
                          for h in model.accommodations)
        rest_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating']
                         for d in model.days for r in model.restaurants)
        return - (attr_score + hotel_score + rest_score)  # 转为最小化问题

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    
    # 获取选中的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取选中的火车
    selected_train_departure = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            selected_train_departure = model.train_departure_data[t]
            break
    
    selected_train_back = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            selected_train_back = model.train_back_data[t]
            break
    
    plan['hotel'] = {
        'name': selected_hotel['name'],
        'cost_per_night': selected_hotel['cost'],
        'total_cost': selected_hotel['cost'] * (travel_days - 1),
        'rating': selected_hotel['rating']
    }
    
    plan['transport'] = {
        'departure': {
            'train_number': selected_train_departure['train_number'],
            'cost': selected_train_departure['cost'],
            'duration': selected_train_departure['duration']
        },
        'return': {
            'train_number': selected_train_back['train_number'],
            'cost': selected_train_back['cost'],
            'duration': selected_train_back['duration']
        }
    }
    
    plan['daily_plans'] = []
    for d in model.days:
        daily_plan = {}
        
        # 获取当天景点
        selected_attr = None
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                selected_attr = model.attr_data[a]
                break
        
        # 获取当天餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append(model.rest_data[r])
        
        # 获取交通方式
        trans_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "打车"
        
        # 计算交通时间和费用
        if selected_attr and selected_hotel:
            if trans_mode == "打车":
                trans_time = (get_trans_params(intra_city_trans, selected_hotel['id'], selected_attr['id'], 'taxi_duration') +
                             get_trans_params(intra_city_trans, selected_attr['id'], selected_hotel['id'], 'taxi_duration'))
                trans_cost = (get_trans_params(intra_city_trans, selected_hotel['id'], selected_attr['id'], 'taxi_cost') +
                             get_trans_params(intra_city_trans, selected_attr['id'], selected_hotel['id'], 'taxi_cost'))
            else:
                trans_time = (get_trans_params(intra_city_trans, selected_hotel['id'], selected_attr['id'], 'bus_duration') +
                             get_trans_params(intra_city_trans, selected_attr['id'], selected_hotel['id'], 'bus_duration'))
                trans_cost = (get_trans_params(intra_city_trans, selected_hotel['id'], selected_attr['id'], 'bus_cost') +
                             get_trans_params(intra_city_trans, selected_attr['id'], selected_hotel['id'], 'bus_cost')) * peoples
        
        daily_plan['day'] = d
        daily_plan['attraction'] = {
            'name': selected_attr['name'],
            'cost': selected_attr['cost'],
            'duration': selected_attr['duration'],
            'rating': selected_attr['rating']
        }
        daily_plan['restaurants'] = [
            {
                'name': r['name'],
                'cost': r['cost'],
                'duration': r['duration'],
                'rating': r['rating'],
                'recommended_food': r['recommended_food']
            } for r in selected_rests
        ]
        daily_plan['transport'] = {
            'mode': trans_mode,
            'time': trans_time,
            'cost': trans_cost
        }
        
        plan['daily_plans'].append(daily_plan)
    
    return json.dumps(plan, indent=2, ensure_ascii=False)

# 主程序
cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
solver = pyo.SolverFactory('scip')
results = solver.solve(model)

plan = generate_daily_plan(model, intra_city_trans)
print(f"```generated_plan\n{plan}\n```")