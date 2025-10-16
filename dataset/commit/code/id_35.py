import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "广州市"
destination_city = "北京市"
budget = 25000
start_date = "2025年04月15日"
end_date = "2025年04月19日"
travel_days = 5
peoples = 2

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    poi_data = {
        'attractions': requests.get(url + f"/attractions/{destination_city}").json(),
        'accommodations': requests.get(url + f"/accommodations/{destination_city}").json(),
        'restaurants': requests.get(url + f"/restaurants/{destination_city}").json()
    }

    # 过滤住宿条件
    poi_data['accommodations'] = [h for h in poi_data['accommodations'] 
                                 if h['rating'] >= 4.7 and h['cost'] <= 850]
    
    # 过滤餐厅条件（人均消费300以内）
    poi_data['restaurants'] = [r for r in poi_data['restaurants']
                              if r['cost'] <= 300]
    
    intra_city_trans = requests.get(url + f"/intra-city-transport/{destination_city}").json()
    return cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans

# 获取交通参数
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

# 构建模型
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 0 for taxi, 1 for bus
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

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        total_rating = sum(
            model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants
        ) + sum(
            model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations
        )
        
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
        
        total_cost = (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
            attraction_cost + restaurant_cost + train_departure_cost + train_back_cost)
        
        return -total_rating + 0.0001 * total_cost  # 最大化评分为主，成本最小化为辅

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

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

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # 每人每天3个餐厅
    def restaurant_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.restaurant_per_day = pyo.Constraint(model.days, rule=restaurant_per_day_rule)

    # 每天1个景点
    def attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.attraction_per_day = pyo.Constraint(model.days, rule=attraction_per_day_rule)

    # 景点不重复
    def no_repeat_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.no_repeat_attraction = pyo.Constraint(model.attractions, rule=no_repeat_attraction_rule)

    # 餐厅不重复
    def no_repeat_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.no_repeat_restaurant = pyo.Constraint(model.restaurants, rule=no_repeat_restaurant_rule)

    # 选择1个出发车次
    def select_one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.select_one_train_departure = pyo.Constraint(rule=select_one_train_departure_rule)

    # 选择1个返程车次
    def select_one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.select_one_train_back = pyo.Constraint(rule=select_one_train_back_rule)

    # 选择1个酒店
    def select_one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.select_one_hotel = pyo.Constraint(rule=select_one_hotel_rule)

    # 每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                        for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] 
                        for r in model.restaurants)
        transport_time = sum(
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
            for h in model.accommodations)
        return attr_time + rest_time + transport_time <= 840
    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = {
                'id': model.hotel_data[h]['id'],
                'name': model.hotel_data[h]['name'],
                'cost': model.hotel_data[h]['cost'],
                'rating': model.hotel_data[h]['rating']
            }
            break
    
    # 获取出发和返程车次
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration'],
                'origin_station': model.train_departure_data[t]['origin_station'],
                'destination_station': model.train_departure_data[t]['destination_station']
            }
            break
    
    back_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            back_train = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration'],
                'origin_station': model.train_back_data[t]['origin_station'],
                'destination_station': model.train_back_data[t]['destination_station']
            }
            break
    
    # 生成每日计划
    for d in model.days:
        day_plan = {
            'hotel': selected_hotel if d != travel_days else None,  # 最后一天不住宿
            'attractions': [],
            'restaurants': [],
            'transport_mode': None
        }
        
        # 获取景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attractions'].append({
                    'id': model.attr_data[a]['id'],
                    'name': model.attr_data[a]['name'],
                    'cost': model.attr_data[a]['cost'],
                    'duration': model.attr_data[a]['duration'],
                    'rating': model.attr_data[a]['rating']
                })
                break
        
        # 获取餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append({
                    'id': model.rest_data[r]['id'],
                    'name': model.rest_data[r]['name'],
                    'cost': model.rest_data[r]['cost'],
                    'duration': model.rest_data[r]['duration'],
                    'rating': model.rest_data[r]['rating'],
                    'recommended_food': model.rest_data[r]['recommended_food']
                })
        
        # 获取交通方式
        transport_mode = 'bus' if pyo.value(model.trans_mode[d]) > 0.5 else 'taxi'
        if d == travel_days:  # 最后一天只有单程
            for a in model.attractions:
                if pyo.value(model.select_attr[d, a]) > 0.5:
                    if transport_mode == 'taxi':
                        cost = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_cost')
                        duration = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_duration')
                    else:
                        cost = peoples * get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_cost')
                        duration = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_duration')
                    day_plan['transport'] = {
                        'mode': transport_mode,
                        'cost': cost,
                        'duration': duration
                    }
                    break
        else:
            for a in model.attractions:
                if pyo.value(model.select_attr[d, a]) > 0.5:
                    if transport_mode == 'taxi':
                        cost = (get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_cost') + 
                               get_trans_params(intra_city_trans, a, selected_hotel['id'], 'taxi_cost'))
                        duration = (get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_duration') + 
                                    get_trans_params(intra_city_trans, a, selected_hotel['id'], 'taxi_duration'))
                    else:
                        cost = peoples * (get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_cost') + 
                                     get_trans_params(intra_city_trans, a, selected_hotel['id'], 'bus_cost'))
                        duration = (get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_duration') + 
                                    get_trans_params(intra_city_trans, a, selected_hotel['id'], 'bus_duration'))
                    day_plan['transport'] = {
                        'mode': transport_mode,
                        'cost': cost,
                        'duration': duration
                    }
                    break
        
        plan[f'Day {d}'] = day_plan
    
    # 添加交通信息
    plan['Transportation'] = {
        'Departure': departure_train,
        'Return': back_train
    }
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    # 求解
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No feasible solution found.")