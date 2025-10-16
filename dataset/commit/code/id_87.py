import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "武汉市"
destination_city = "三亚市"
budget = 9200
start_date = "2025年06月25日"
end_date = "2025年06月28日"
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

    # 过滤评分4.6以上的酒店
    poi_data['accommodations'] = [h for h in poi_data['accommodations'] if h['rating'] >= 4.6]
    
    # 过滤指定景点
    required_attractions = ['千古情', '玫瑰谷']
    poi_data['attractions'] = [a for a in poi_data['attractions'] if a['name'] in required_attractions]
    
    # 过滤海鲜烧烤餐厅且人均消费350元内
    poi_data['restaurants'] = [r for r in poi_data['restaurants'] 
                              if '海鲜烧烤' in r['type'] and r['cost'] <= 350]

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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 0: taxi, 1: bus
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

    # 约束条件：每日活动时间不超过840分钟
    def activity_time_rule(model, d):
        if d == travel_days:  # 最后一天不计算住宿
            return sum(
                model.select_attr[d, a] * model.attr_data[a]['duration'] +
                sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
                    for r in model.restaurants) +
                sum(model.attr_hotel[d, a, h] * (
                    (1 - model.trans_mode[d]) * get_trans_params(intra_city_trans, h, a, 'taxi_duration') +
                    model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_duration')
                ) for a in model.attractions for h in model.accommodations)
                for a in model.attractions
            ) <= 840
        else:
            return sum(
                model.select_attr[d, a] * model.attr_data[a]['duration'] +
                sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
                    for r in model.restaurants) +
                sum(model.attr_hotel[d, a, h] * 2 * (  # 往返交通
                    (1 - model.trans_mode[d]) * get_trans_params(intra_city_trans, h, a, 'taxi_duration') +
                    model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_duration')
                ) for a in model.attractions for h in model.accommodations)
                for a in model.attractions
            ) <= 840

    model.activity_time_constraint = pyo.Constraint(model.days, rule=activity_time_rule)

    # 约束条件：每日必须选择1个景点
    def attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.attraction_constraint = pyo.Constraint(model.days, rule=attraction_rule)

    # 约束条件：景点不重复
    def attraction_unique_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.attraction_unique_constraint = pyo.Constraint(model.attractions, rule=attraction_unique_rule)

    # 约束条件：每日选择3个餐厅
    def restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.restaurant_constraint = pyo.Constraint(model.days, rule=restaurant_rule)

    # 约束条件：餐厅不重复
    def restaurant_unique_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.restaurant_unique_constraint = pyo.Constraint(model.restaurants, rule=restaurant_unique_rule)

    # 约束条件：选择1个酒店
    def hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.hotel_constraint = pyo.Constraint(rule=hotel_rule)

    # 约束条件：选择1个去程火车
    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.train_departure_constraint = pyo.Constraint(rule=train_departure_rule)

    # 约束条件：选择1个返程火车
    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.train_back_constraint = pyo.Constraint(rule=train_back_rule)

    # 约束条件：以公交为主
    def bus_preference_rule(model, d):
        return model.trans_mode[d] == 1

    model.bus_preference_constraint = pyo.Constraint(model.days, rule=bus_preference_rule)

    # 目标函数：最大化评分
    def obj_rule(model):
        attraction_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                              for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] 
                          for h in model.accommodations) * (travel_days - 1)
        restaurant_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                              for d in model.days for r in model.restaurants)
        return - (attraction_score + hotel_score + restaurant_score)  # 最小化负分即最大化评分

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
                model.trans_mode[d] * (
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
        return (hotel_cost + transport_cost + attraction_cost + restaurant_cost + 
                train_departure_cost + train_back_cost) <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    days = sorted(list(model.days))
    attractions = list(model.attractions)
    restaurants = list(model.restaurants)
    accommodations = list(model.accommodations)
    
    selected_hotel = None
    for h in accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = h
            break
    
    selected_train_departure = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            selected_train_departure = t
            break
    
    selected_train_back = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            selected_train_back = t
            break
    
    plan = {
        "origin_city": origin_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "peoples": peoples,
        "train_departure": {
            "train_number": model.train_departure_data[selected_train_departure]['train_number'],
            "cost": pyo.value(model.train_departure_data[selected_train_departure]['cost']),
            "duration": pyo.value(model.train_departure_data[selected_train_departure]['duration']),
            "origin_station": model.train_departure_data[selected_train_departure]['origin_station'],
            "destination_station": model.train_departure_data[selected_train_departure]['destination_station']
        },
        "train_back": {
            "train_number": model.train_back_data[selected_train_back]['train_number'],
            "cost": pyo.value(model.train_back_data[selected_train_back]['cost']),
            "duration": pyo.value(model.train_back_data[selected_train_back]['duration']),
            "origin_station": model.train_back_data[selected_train_back]['origin_station'],
            "destination_station": model.train_back_data[selected_train_back]['destination_station']
        },
        "hotel": {
            "name": model.hotel_data[selected_hotel]['name'],
            "cost_per_night": pyo.value(model.hotel_data[selected_hotel]['cost']),
            "rating": pyo.value(model.hotel_data[selected_hotel]['rating']),
            "total_cost": pyo.value(model.hotel_data[selected_hotel]['cost']) * (travel_days - 1)
        },
        "daily_plans": []
    }
    
    for d in days:
        daily_plan = {
            "day": d,
            "attraction": None,
            "restaurants": [],
            "transport_mode": "bus" if pyo.value(model.trans_mode[d]) > 0.5 else "taxi"
        }
        
        for a in attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                daily_plan["attraction"] = {
                    "name": model.attr_data[a]['name'],
                    "cost": pyo.value(model.attr_data[a]['cost']),
                    "duration": pyo.value(model.attr_data[a]['duration']),
                    "rating": pyo.value(model.attr_data[a]['rating'])
                }
                break
        
        for r in restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                daily_plan["restaurants"].append({
                    "name": model.rest_data[r]['name'],
                    "cost": pyo.value(model.rest_data[r]['cost']),
                    "duration": pyo.value(model.rest_data[r]['duration']),
                    "queue_time": pyo.value(model.rest_data[r]['queue_time']),
                    "rating": pyo.value(model.rest_data[r]['rating']),
                    "recommended_food": model.rest_data[r]['recommended_food']
                })
        
        plan["daily_plans"].append(daily_plan)
    
    return plan

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()