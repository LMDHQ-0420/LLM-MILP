import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "厦门市"
destination_city = "北京市"
budget = 7500
start_date = "2025年05月28日"
end_date = "2025年05月31日"
travel_days = 4
peoples = 1

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

    # 过滤评分4.6以上的特色民宿
    poi_data['accommodations'] = [h for h in poi_data['accommodations'] 
                                 if h['rating'] >= 4.6 and h['type'] == '特色民宿']
    
    # 过滤必须包含的景点
    required_attrs = ['天坛', '南锣鼓巷']
    poi_data['attractions'] = [a for a in poi_data['attractions'] 
                              if a['name'] in required_attrs or a['name'] not in required_attrs]
    
    # 过滤推荐炸酱面且人均300以内的餐厅
    poi_data['restaurants'] = [r for r in poi_data['restaurants'] 
                              if '炸酱面' in r['recommended_food'] and r['cost'] <= 300]

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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 默认选择公交
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

    # 必须包含的景点约束
    def required_attr_rule(model):
        required_attrs = ['天坛', '南锣鼓巷']
        required_attr_ids = [a['id'] for a in poi_data['attractions'] if a['name'] in required_attrs]
        return sum(model.select_attr[d, a] for d in model.days for a in required_attr_ids) >= len(required_attrs)

    model.required_attr_constraint = pyo.Constraint(rule=required_attr_rule)

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        # 计算总评分（负值，因为我们要最小化）
        total_rating = -(
            sum(model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions) +
            sum(model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations) +
            sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants)
        ) / (travel_days * 4)  # 归一化

        # 计算总成本
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                         for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                              for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                              for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
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
        
        total_cost = hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost
        
        # 加权组合目标（评分占60%，成本占40%）
        return 0.6 * total_rating + 0.4 * total_cost / budget

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
        return hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # 每日活动时间约束
    def daily_time_rule(model, d):
        if d == 1:  # 第一天不考虑火车时间
            attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
            rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                           for r in model.restaurants)
            trans_time = sum(
                model.attr_hotel[d, a, h] * (
                    model.trans_mode[d] * (
                        get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                        get_trans_params(intra_city_trans, a, h, 'bus_duration')
                    )
                )
                for a in model.attractions
                for h in model.accommodations
            )
            return attr_time + rest_time + trans_time <= 840
        elif d == travel_days:  # 最后一天
            attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
            rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                           for r in model.restaurants)
            trans_time = sum(
                model.attr_hotel[d, a, h] * (
                    model.trans_mode[d] * (
                        get_trans_params(intra_city_trans, h, a, 'bus_duration')
                    )
                )
                for a in model.attractions
                for h in model.accommodations
            )
            return attr_time + rest_time + trans_time <= 840
        else:  # 中间天数
            attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
            rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                           for r in model.restaurants)
            trans_time = sum(
                model.attr_hotel[d, a, h] * (
                    model.trans_mode[d] * (
                        get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                        get_trans_params(intra_city_trans, a, h, 'bus_duration')
                    )
                )
                for a in model.attractions
                for h in model.accommodations
            )
            return attr_time + rest_time + trans_time <= 840

    model.daily_time_constraint = pyo.Constraint(model.days, rule=daily_time_rule)

    # 每日必须选择1个景点
    def daily_attr_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.daily_attr_constraint = pyo.Constraint(model.days, rule=daily_attr_rule)

    # 景点不重复
    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attr_constraint = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 每日必须选择3个餐厅
    def daily_rest_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.daily_rest_constraint = pyo.Constraint(model.days, rule=daily_rest_rule)

    # 餐厅不重复
    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_rest_constraint = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 必须选择1个酒店
    def hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.hotel_constraint = pyo.Constraint(rule=hotel_rule)

    # 必须选择1个去程火车
    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.train_departure_constraint = pyo.Constraint(rule=train_departure_rule)

    # 必须选择1个返程火车
    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.train_back_constraint = pyo.Constraint(rule=train_back_rule)

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
    
    # 获取选择的火车
    selected_train_departure = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            selected_train_departure = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration'],
                'origin_station': model.train_departure_data[t]['origin_station'],
                'destination_station': model.train_departure_data[t]['destination_station']
            }
            break
    
    selected_train_back = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            selected_train_back = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration'],
                'origin_station': model.train_back_data[t]['origin_station'],
                'destination_station': model.train_back_data[t]['destination_station']
            }
            break
    
    # 生成每日计划
    for d in model.days:
        daily_plan = {
            'hotel': selected_hotel if d < travel_days else None,
            'attractions': [],
            'restaurants': [],
            'transport_mode': '公交' if pyo.value(model.trans_mode[d]) > 0.5 else '出租车'
        }
        
        # 添加景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                daily_plan['attractions'].append({
                    'id': model.attr_data[a]['id'],
                    'name': model.attr_data[a]['name'],
                    'cost': model.attr_data[a]['cost'],
                    'duration': model.attr_data[a]['duration'],
                    'rating': model.attr_data[a]['rating']
                })
                break  # 每天只选一个景点
        
        # 添加餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                daily_plan['restaurants'].append({
                    'id': model.rest_data[r]['id'],
                    'name': model.rest_data[r]['name'],
                    'cost': model.rest_data[r]['cost'],
                    'recommended_food': model.rest_data[r]['recommended_food'],
                    'rating': model.rest_data[r]['rating']
                })
                if len(daily_plan['restaurants']) == 3:
                    break  # 每天选3个餐厅
        
        # 计算交通时间
        if d < travel_days and daily_plan['attractions']:
            attr_id = daily_plan['attractions'][0]['id']
            hotel_id = selected_hotel['id']
            key1 = f"{hotel_id},{attr_id}"
            key2 = f"{attr_id},{hotel_id}"
            
            if key1 in intra_city_trans:
                trans_data = intra_city_trans[key1]
                if daily_plan['transport_mode'] == '公交':
                    daily_plan['transport_time'] = trans_data['bus_duration']
                    daily_plan['transport_cost'] = trans_data['bus_cost']
                else:
                    daily_plan['transport_time'] = trans_data['taxi_duration']
                    daily_plan['transport_cost'] = trans_data['taxi_cost']
            
            if key2 in intra_city_trans and d < travel_days:
                trans_data = intra_city_trans[key2]
                if 'transport_time' in daily_plan:
                    daily_plan['transport_time'] += trans_data['bus_duration'] if daily_plan['transport_mode'] == '公交' else trans_data['taxi_duration']
                    daily_plan['transport_cost'] += trans_data['bus_cost'] if daily_plan['transport_mode'] == '公交' else trans_data['taxi_cost']
        
        plan[f"Day {d}"] = daily_plan
    
    # 添加火车信息
    plan['train_departure'] = selected_train_departure
    plan['train_back'] = selected_train_back
    
    return json.dumps(plan, indent=2, ensure_ascii=False)

# 主函数
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()