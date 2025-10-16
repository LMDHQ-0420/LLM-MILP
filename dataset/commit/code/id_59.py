import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "成都市"
destination_city = "苏州市"
budget = 6800
start_date = "2025年06月15日"
end_date = "2025年06月18日"
travel_days = 4
peoples = 1

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 过滤景点：只包含唐寅园、石湖
    all_attractions = requests.get(url + f"/attractions/{destination_city}").json()
    filtered_attractions = [a for a in all_attractions if a['name'] in ['唐寅园', '石湖']]
    
    # 过滤酒店：评分4.6以上
    all_hotels = requests.get(url + f"/accommodations/{destination_city}").json()
    filtered_hotels = [h for h in all_hotels if float(h['rating']) >= 4.6]
    
    # 过滤餐厅：推荐清炒虾仁且人均消费400以内
    all_restaurants = requests.get(url + f"/restaurants/{destination_city}").json()
    filtered_restaurants = [r for r in all_restaurants 
                           if '清炒虾仁' in r['recommended_food'] and float(r['cost']) <= 400]

    poi_data = {
        'attractions': filtered_attractions,
        'accommodations': filtered_hotels,
        'restaurants': filtered_restaurants
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
            'recommended_food': restaurant_dict[r['recommended_food']],
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

    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 默认公交
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)
    model.attr_hotel = pyo.Var(model.days, model.attractions, model.accommodations, domain=pyo.Binary, initialize=0, bounds=(0, 1))

    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]
    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]
    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

    model.link_attr_hotel1 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule1)
    model.link_attr_hotel2 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule2)
    model.link_attr_hotel3 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule3)

    # 目标函数：最小化交通耗时
    def obj_rule(model):
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
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )
        return transport_time

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # 约束条件
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)

    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)

    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)

    def hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.hotel_selection = pyo.Constraint(rule=hotel_rule)

    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.train_departure_selection = pyo.Constraint(rule=train_departure_rule)

    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.train_back_selection = pyo.Constraint(rule=train_back_rule)

    def daily_time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
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
        return attr_time + rest_time + transport_time <= 840
    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants)
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
            for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        return hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost <= budget
    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for day in model.days:
        daily_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=day-1)).strftime("%Y年%m月%d日"),
            'attractions': [],
            'restaurants': [],
            'accommodation': None,
            'transport_mode': None
        }
        
        # 景点
        for attr in model.attractions:
            if pyo.value(model.select_attr[day, attr]) > 0.5:
                daily_plan['attractions'].append({
                    'name': model.attr_data[attr]['name'],
                    'duration': model.attr_data[attr]['duration'],
                    'cost': model.attr_data[attr]['cost']
                })
        
        # 餐厅
        for rest in model.restaurants:
            if pyo.value(model.select_rest[day, rest]) > 0.5:
                daily_plan['restaurants'].append({
                    'name': model.rest_data[rest]['name'],
                    'duration': model.rest_data[rest]['duration'],
                    'cost': model.rest_data[rest]['cost'],
                    'recommended_food': model.rest_data[rest]['recommended_food']
                })
        
        # 住宿
        if day < travel_days:  # 最后一天不安排住宿
            for hotel in model.accommodations:
                if pyo.value(model.select_hotel[hotel]) > 0.5:
                    daily_plan['accommodation'] = {
                        'name': model.hotel_data[hotel]['name'],
                        'cost': model.hotel_data[hotel]['cost'],
                        'rating': model.hotel_data[hotel]['rating']
                    }
        
        # 交通方式
        trans_mode = "公交" if pyo.value(model.trans_mode[day]) > 0.5 else "打车"
        daily_plan['transport_mode'] = trans_mode
        
        plan[f"Day {day}"] = daily_plan
    
    # 添加火车信息
    for train in model.train_departure:
        if pyo.value(model.select_train_departure[train]) > 0.5:
            plan['departure_train'] = {
                'train_number': model.train_departure_data[train]['train_number'],
                'departure_station': model.train_departure_data[train]['origin_station'],
                'arrival_station': model.train_departure_data[train]['destination_station'],
                'duration': model.train_departure_data[train]['duration'],
                'cost': model.train_departure_data[train]['cost']
            }
    
    for train in model.train_back:
        if pyo.value(model.select_train_back[train]) > 0.5:
            plan['return_train'] = {
                'train_number': model.train_back_data[train]['train_number'],
                'departure_station': model.train_back_data[train]['origin_station'],
                'arrival_station': model.train_back_data[train]['destination_station'],
                'duration': model.train_back_data[train]['duration'],
                'cost': model.train_back_data[train]['cost']
            }
    
    return json.dumps(plan, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No feasible solution found")