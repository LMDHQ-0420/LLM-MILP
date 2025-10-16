import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "厦门市"
destination_city = "北京市"
budget = 18000
start_date = "2025年10月08日"
end_date = "2025年10月12日"
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

    # 过滤景点和酒店
    attractions = [a for a in poi_data['attractions'] if a['id'] in ['故宫', '颐和园']]
    hotels = [h for h in poi_data['accommodations'] if float(h['rating']) >= 4.8]
    restaurants = [r for r in poi_data['restaurants'] if '烤鸭' in r['recommended_food'] and float(r['cost']) <= 350]

    attraction_dict = {a['id']: a for a in attractions}
    hotel_dict = {h['id']: h for h in hotels}
    restaurant_dict = {r['id']: r for r in restaurants}
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 1表示地铁
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

    # 每日必须选择一个景点
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)

    # 景点不重复
    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)

    # 每日必须选择三个餐厅
    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    # 餐厅不重复
    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)

    # 只能选择一个酒店
    def single_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.single_hotel = pyo.Constraint(rule=single_hotel_rule)

    # 只能选择一个去程和一个返程车次
    def single_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def single_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.single_train_departure = pyo.Constraint(rule=single_train_departure_rule)
    model.single_train_back = pyo.Constraint(rule=single_train_back_rule)

    # 每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
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

    # 目标函数：最小化交通时间（优先选择高铁）
    def obj_rule(model):
        train_departure_time = sum(model.select_train_departure[t] * model.train_departure_data[t]['duration']
                                   for t in model.train_departure)
        train_back_time = sum(model.select_train_back[t] * model.train_back_data[t]['duration']
                              for t in model.train_back)
        return train_departure_time + train_back_time

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            "date": (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            "attractions": [],
            "restaurants": [],
            "accommodation": None,
            "transport": []
        }
        
        # 景点
        for a in model.attractions:
            if model.select_attr[d, a].value > 0.5:
                day_plan["attractions"].append({
                    "id": model.attr_data[a]['id'],
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration'],
                    "cost": model.attr_data[a]['cost']
                })
        
        # 餐厅
        for r in model.restaurants:
            if model.select_rest[d, r].value > 0.5:
                day_plan["restaurants"].append({
                    "id": model.rest_data[r]['id'],
                    "name": model.rest_data[r]['name'],
                    "duration": model.rest_data[r]['duration'],
                    "cost": model.rest_data[r]['cost'],
                    "recommended_food": model.rest_data[r]['recommended_food']
                })
        
        # 酒店
        if d < travel_days:
            for h in model.accommodations:
                if model.select_hotel[h].value > 0.5:
                    day_plan["accommodation"] = {
                        "id": model.hotel_data[h]['id'],
                        "name": model.hotel_data[h]['name'],
                        "cost": model.hotel_data[h]['cost'],
                        "rating": model.hotel_data[h]['rating']
                    }
        
        # 交通
        for a in model.attractions:
            for h in model.accommodations:
                if model.attr_hotel[d, a, h].value > 0.5:
                    mode = "地铁" if model.trans_mode[d].value > 0.5 else "出租车"
                    cost = get_trans_params(intra_city_trans, h, a, 'bus_cost' if mode == "地铁" else 'taxi_cost')
                    duration = get_trans_params(intra_city_trans, h, a, 'bus_duration' if mode == "地铁" else 'taxi_duration')
                    day_plan["transport"].append({
                        "from": model.hotel_data[h]['name'] if d < travel_days else "前一晚酒店",
                        "to": model.attr_data[a]['name'],
                        "mode": mode,
                        "cost": cost,
                        "duration": duration
                    })
                    day_plan["transport"].append({
                        "from": model.attr_data[a]['name'],
                        "to": model.hotel_data[h]['name'] if d < travel_days else "返程",
                        "mode": mode,
                        "cost": cost,
                        "duration": duration
                    })
        
        plan[f"day{d}"] = day_plan
    
    # 添加火车信息
    for t in model.train_departure:
        if model.select_train_departure[t].value > 0.5:
            plan["departure_train"] = {
                "train_number": model.train_departure_data[t]['train_number'],
                "origin": model.train_departure_data[t]['origin_station'],
                "destination": model.train_departure_data[t]['destination_station'],
                "duration": model.train_departure_data[t]['duration'],
                "cost": model.train_departure_data[t]['cost']
            }
    
    for t in model.train_back:
        if model.select_train_back[t].value > 0.5:
            plan["return_train"] = {
                "train_number": model.train_back_data[t]['train_number'],
                "origin": model.train_back_data[t]['origin_station'],
                "destination": model.train_back_data[t]['destination_station'],
                "duration": model.train_back_data[t]['duration'],
                "cost": model.train_back_data[t]['cost']
            }
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")