import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "深圳市"
destination_city = "洛阳市"
budget = 22000
start_date = "2025年08月22日"
end_date = "2025年08月26日"
travel_days = 5
peoples = 3

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

    # 筛选景点：必须包含龙门石窟和倒盏民俗村
    all_attractions = {a['id']: a for a in poi_data['attractions']}
    required_attractions = [a for a in poi_data['attractions'] if a['name'] in ['龙门石窟', '倒盏民俗村']]
    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    
    # 筛选酒店：评分4.7以上
    hotel_dict = {h['id']: h for h in poi_data['accommodations'] if float(h['rating']) >= 4.7}
    
    # 筛选餐厅：包含水席宴且人均消费<=160
    restaurant_dict = {r['id']: r for r in poi_data['restaurants'] 
                      if float(r['cost']) <= 160 and ('水席' in r['name'] or '水席' in r['recommended_food'])}
    
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)
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

    # 必须包含指定景点
    def required_attractions_rule(model):
        required_attraction_ids = [a['id'] for a in required_attractions]
        return sum(model.select_attr[d, a] for d in model.days for a in model.attractions if a in required_attraction_ids) == len(required_attraction_ids)

    model.required_attractions = pyo.Constraint(rule=required_attractions_rule)

    # 约束条件：每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        if d == travel_days:  # 最后一天不计算往返交通
            transport_time = 0
        else:
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
                for h in model.accommodations
            )
        
        attraction_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        restaurant_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
        
        return attraction_time + restaurant_time + transport_time <= 840

    model.daily_time_constraint = pyo.Constraint(model.days, rule=daily_time_rule)

    # 约束条件：每天1个景点
    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    # 约束条件：每天3个餐厅
    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    # 约束条件：景点不重复
    def unique_attractions_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions_rule)

    # 约束条件：餐厅不重复
    def unique_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants_rule)

    # 约束条件：选择1个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 约束条件：选择1个去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    # 约束条件：选择1个返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 目标函数：最大化景点参观时间，最小化排队时间
    def obj_rule(model):
        attraction_duration = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                                for d in model.days for a in model.attractions)
        queue_time = sum(model.select_rest[d, r] * model.rest_data[r]['queue_time'] 
                        for d in model.days for r in model.restaurants)
        return -attraction_duration + queue_time

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
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
            ) + \
                    model.trans_mode[d] * peoples * (
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
        return (peoples+1)//2 * hotel_cost + peoples * (attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) + transport_cost <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'attractions': [],
            'restaurants': [],
            'hotel': None,
            'transport_mode': None,
            'transport_time': 0,
            'total_time': 0
        }
        
        # 获取景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attr_info = model.attr_data[a]
                day_plan['attractions'].append({
                    'name': attr_info['name'],
                    'duration': attr_info['duration'],
                    'cost': attr_info['cost']
                })
        
        # 获取餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                rest_info = model.rest_data[r]
                day_plan['restaurants'].append({
                    'name': rest_info['name'],
                    'duration': rest_info['duration'],
                    'queue_time': rest_info['queue_time'],
                    'cost': rest_info['cost'],
                    'recommended_food': rest_info['recommended_food']
                })
        
        # 获取酒店（最后一天没有）
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    hotel_info = model.hotel_data[h]
                    day_plan['hotel'] = {
                        'name': hotel_info['name'],
                        'cost': hotel_info['cost'],
                        'rating': hotel_info['rating']
                    }
        
        # 获取交通方式和时间
        transport_mode = "taxi" if pyo.value(model.trans_mode[d]) < 0.5 else "bus"
        day_plan['transport_mode'] = transport_mode
        
        if d < travel_days and day_plan['attractions'] and day_plan['hotel']:
            attr_id = day_plan['attractions'][0]['name']
            hotel_id = day_plan['hotel']['name']
            key1 = f"{hotel_id},{attr_id}"
            key2 = f"{attr_id},{hotel_id}"
            
            if key1 in intra_city_trans or key2 in intra_city_trans:
                key = key1 if key1 in intra_city_trans else key2
                transport_data = intra_city_trans[key]
                if transport_mode == "taxi":
                    transport_time = float(transport_data['taxi_duration']) * 2
                else:
                    transport_time = float(transport_data['bus_duration']) * 2
                day_plan['transport_time'] = transport_time
        
        # 计算总时间
        attraction_time = sum(a['duration'] for a in day_plan['attractions'])
        restaurant_time = sum(r['duration'] + r['queue_time'] for r in day_plan['restaurants'])
        day_plan['total_time'] = attraction_time + restaurant_time + day_plan['transport_time']
        
        plan[f"Day {d}"] = day_plan
    
    # 添加火车信息
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            train_info = model.train_departure_data[t]
            plan['departure_train'] = {
                'train_number': train_info['train_number'],
                'origin_station': train_info['origin_station'],
                'destination_station': train_info['destination_station'],
                'duration': train_info['duration'],
                'cost': train_info['cost']
            }
    
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            train_info = model.train_back_data[t]
            plan['return_train'] = {
                'train_number': train_info['train_number'],
                'origin_station': train_info['origin_station'],
                'destination_station': train_info['destination_station'],
                'duration': train_info['duration'],
                'cost': train_info['cost']
            }
    
    return plan

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()