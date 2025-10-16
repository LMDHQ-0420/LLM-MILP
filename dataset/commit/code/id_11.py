import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "厦门市"
destination_city = "青岛市"
budget = 0  # 不限制预算
start_date = "2025年07月05日"
end_date = "2025年07月09日"
travel_days = 5
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

    # 过滤景点：只包含栈桥、八大关、崂山等经典景点且门票300元以内
    filtered_attractions = [a for a in poi_data['attractions'] 
                           if a['name'] in ['栈桥', '八大关', '崂山'] and float(a['cost']) <= 300]
    poi_data['attractions'] = filtered_attractions

    # 过滤酒店：评分4.6以上、价格低于800元的经济型连锁酒店
    filtered_hotels = [h for h in poi_data['accommodations'] 
                      if float(h['rating']) >= 4.6 and float(h['cost']) <= 800 and '经济型' in h['type']]
    poi_data['accommodations'] = filtered_hotels

    # 过滤餐厅：推荐地道海鲜、本地特色，人均消费不超400元
    filtered_restaurants = [r for r in poi_data['restaurants'] 
                           if ('海鲜' in r['type'] or '本地特色' in r['type']) and float(r['cost']) <= 400]
    poi_data['restaurants'] = filtered_restaurants

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
            'recommended_food': restaurant_dict[r['recommended_food'] if 'recommended_food' in r else ''],
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
            'destination_id': train_back_dict[t['destination_id']],
            'destination_station': train_back_dict[t['destination_station']]
        }
    )

    # 定义变量
    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 0-taxi, 1-bus
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

    # 约束条件：每日必须选择1个景点且不重复
    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    def unique_attractions_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions_rule)

    # 约束条件：每日必须选择3个餐厅且不重复
    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    def unique_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants_rule)

    # 约束条件：必须选择1个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 约束条件：必须选择1个去程火车和1个返程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 约束条件：每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        attraction_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        restaurant_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
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
        return attraction_time + restaurant_time + transport_time <= 840

    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        total_rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions) + \
                      sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants) + \
                      sum(model.select_hotel[h] * model.hotel_data[h]['rating'] * (travel_days - 1) for h in model.accommodations)
        
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants)
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
            for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        
        total_cost = (peoples+1)//2 * hotel_cost + transport_cost + peoples * (
            attraction_cost + restaurant_cost + train_departure_cost + train_back_cost
        )
        
        return -total_rating + 0.001 * total_cost  # 主要优化评分，次要优化成本

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {
        "origin_city": origin_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "travel_days": travel_days,
        "peoples": peoples,
        "total_cost": 0,
        "daily_plans": []
    }

    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break

    # 获取选择的火车
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

    # 计算总成本
    hotel_cost = selected_hotel['cost'] * (travel_days - 1)
    train_departure_cost = selected_train_departure['cost'] * peoples
    train_back_cost = selected_train_back['cost'] * peoples

    attraction_cost = 0
    restaurant_cost = 0
    transport_cost = 0

    for d in model.days:
        daily_plan = {
            "day": d,
            "date": (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            "attractions": [],
            "restaurants": [],
            "hotel": None,
            "transport": {
                "train": None,
                "intra_city": []
            }
        }

        if d == 1:
            daily_plan["transport"]["train"] = {
                "type": "departure",
                "train_number": selected_train_departure['train_number'],
                "departure_station": selected_train_departure['origin_station'],
                "arrival_station": selected_train_departure['destination_station'],
                "departure_time": "上午",
                "duration": selected_train_departure['duration'],
                "cost": selected_train_departure['cost']
            }

        if d == travel_days:
            daily_plan["transport"]["train"] = {
                "type": "return",
                "train_number": selected_train_back['train_number'],
                "departure_station": selected_train_back['origin_station'],
                "arrival_station": selected_train_back['destination_station'],
                "departure_time": "下午",
                "duration": selected_train_back['duration'],
                "cost": selected_train_back['cost']
            }

        # 添加景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attr_info = model.attr_data[a]
                daily_plan["attractions"].append({
                    "name": attr_info['name'],
                    "cost": attr_info['cost'],
                    "duration": attr_info['duration'],
                    "rating": attr_info['rating']
                })
                attraction_cost += attr_info['cost']
                break

        # 添加餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                rest_info = model.rest_data[r]
                daily_plan["restaurants"].append({
                    "name": rest_info['name'],
                    "cost": rest_info['cost'],
                    "duration": rest_info['duration'],
                    "rating": rest_info['rating'],
                    "recommended_food": rest_info.get('recommended_food', '')
                })
                restaurant_cost += rest_info['cost']

        # 添加酒店和市内交通
        if d < travel_days:
            daily_plan["hotel"] = {
                "name": selected_hotel['name'],
                "cost": selected_hotel['cost'],
                "rating": selected_hotel['rating'],
                "feature": selected_hotel['feature']
            }

            # 获取市内交通方式
            transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "打车"
            for a in model.attractions:
                if pyo.value(model.select_attr[d, a]) > 0.5:
                    attr_info = model.attr_data[a]
                    if transport_mode == "公交":
                        cost = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_cost') * 2
                        duration = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'bus_duration') * 2
                    else:
                        cost = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_cost') * 2
                        duration = get_trans_params(intra_city_trans, selected_hotel['id'], a, 'taxi_duration') * 2
                    
                    daily_plan["transport"]["intra_city"].append({
                        "from": selected_hotel['name'],
                        "to": attr_info['name'],
                        "mode": transport_mode,
                        "cost": cost,
                        "duration": duration
                    })
                    transport_cost += cost
                    break

        plan["daily_plans"].append(daily_plan)

    total_cost = (peoples+1)//2 * hotel_cost + transport_cost + peoples * (
        attraction_cost + restaurant_cost + train_departure_cost + train_back_cost
    )
    plan["total_cost"] = total_cost

    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No solution found")