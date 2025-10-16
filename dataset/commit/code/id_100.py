import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "深圳市"
destination_city = "广州市"
budget = 8000
start_date = "2025年05月20日"
end_date = "2025年05月23日"
travel_days = 4
peoples = 2

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    poi_data = {
        'attractions': [a for a in requests.get(url + f"/attractions/{destination_city}").json() 
                       if a['type'] == '历史文化'],
        'accommodations': [h for h in requests.get(url + f"/accommodations/{destination_city}").json() 
                          if h['rating'] >= 4.6 and h['cost'] <= 1000 and '连锁' in h['feature']],
        'restaurants': [r for r in requests.get(url + f"/restaurants/{destination_city}").json() 
                       if '早茶' in r['recommended_food']]
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

    def obj_rule(model):
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
        return transport_cost + (peoples+1) // 2 * hotel_cost + peoples * (
                      attraction_cost + restaurant_cost + train_departure_cost + train_back_cost)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

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
        return (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
                     attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    model.budget_constr = pyo.Constraint(rule=budget_rule)

    def daily_activity_time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                        for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
                        for r in model.restaurants)
        transport_time = sum(
            model.attr_hotel[d, a, h] * (
                model.trans_mode[d] * (
                    get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                    get_trans_params(intra_city_trans, a, h, 'bus_duration')
                )
            )
            for a in model.attractions
            for h in model.accommodations)
        return attr_time + rest_time + transport_time <= 840

    model.daily_time_constr = pyo.Constraint(model.days, rule=daily_activity_time_rule)

    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attr_constr = pyo.Constraint(model.days, rule=one_attr_per_day_rule)

    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_rest_constr = pyo.Constraint(model.days, rule=three_rest_per_day_rule)

    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel_constr = pyo.Constraint(rule=one_hotel_rule)

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure_constr = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back_constr = pyo.Constraint(rule=one_train_back_rule)

    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attr_constr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_rest_constr = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    def bus_transport_rule(model, d):
        return model.trans_mode[d] == 1

    model.bus_transport_constr = pyo.Constraint(model.days, rule=bus_transport_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            "date": (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            "attraction": None,
            "restaurants": [],
            "accommodation": None,
            "transport_mode": "公交"
        }
        
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan["attraction"] = {
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration'],
                    "cost": model.attr_data[a]['cost']
                }
                break
        
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan["restaurants"].append({
                    "name": model.rest_data[r]['name'],
                    "cost": model.rest_data[r]['cost'],
                    "recommended_food": model.rest_data[r]['recommended_food']
                })
        
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    day_plan["accommodation"] = {
                        "name": model.hotel_data[h]['name'],
                        "cost": model.hotel_data[h]['cost']
                    }
                    break
        
        plan[f"Day {d}"] = day_plan
    
    train_departure = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            train_departure = {
                "train_number": model.train_departure_data[t]['train_number'],
                "departure_station": model.train_departure_data[t]['origin_station'],
                "arrival_station": model.train_departure_data[t]['destination_station'],
                "duration": model.train_departure_data[t]['duration'],
                "cost": model.train_departure_data[t]['cost']
            }
            break
    
    train_back = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            train_back = {
                "train_number": model.train_back_data[t]['train_number'],
                "departure_station": model.train_back_data[t]['origin_station'],
                "arrival_station": model.train_back_data[t]['destination_station'],
                "duration": model.train_back_data[t]['duration'],
                "cost": model.train_back_data[t]['cost']
            }
            break
    
    plan["Transportation"] = {
        "Departure": train_departure,
        "Return": train_back
    }
    
    total_cost = pyo.value(model.obj)
    plan["Total Cost"] = total_cost
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")