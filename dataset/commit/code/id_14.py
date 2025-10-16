import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "贵阳市"
destination_city = "洛阳市"
budget = 0
start_date = "2025年06月10日"
end_date = "2025年06月12日"
travel_days = 3
peoples = 4

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

    days = list(range(1, travel_days + 1))
    model.days = pyo.Set(initialize=days)

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations'] if float(h['cost']) <= 1000 and '早餐' in h['feature']}
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
            'rating': float(hotel_dict[h]['rating'])
        }
    )

    model.rest_data = pyo.Param(
        model.restaurants,
        initialize=lambda m, r: {
            'id': restaurant_dict[r]['id'],
            'name': restaurant_dict[r]['name'],
            'cost': float(restaurant_dict[r]['cost']),
            'rating': float(restaurant_dict[r]['rating']),
            'duration': float(restaurant_dict[r]['duration'])
        }
    )

    model.train_departure_data = pyo.Param(
        model.train_departure,
        initialize=lambda m, t: {
            'train_number': train_departure_dict[t]['train_number'],
            'cost': float(train_departure_dict[t]['cost']),
            'duration': float(train_departure_dict[t]['duration'])
        }
    )
    model.train_back_data = pyo.Param(
        model.train_back,
        initialize=lambda m, t: {
            'train_number': train_back_dict[t]['train_number'],
            'cost': float(train_back_dict[t]['cost']),
            'duration': float(train_back_dict[t]['duration'])
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
        bounds=(0, 1)
    )

    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]
    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]
    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

    model.link_attr_hotel1 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule1)
    model.link_attr_hotel2 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule2)
    model.link_attr_hotel3 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule3)

    def obj_rule(model):
        total_rating = sum(
            model.select_attr[d, a] * model.attr_data[a]['rating'] * 2 +
            sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for r in model.restaurants) +
            model.select_hotel[h] * model.hotel_data[h]['rating'] * 0.5
            for d in model.days for a in model.attractions for h in model.accommodations
        )
        return -total_rating

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants)
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
            for a in model.attractions for h in model.accommodations
        )
        return attr_time + rest_time + transport_time <= 840
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            "attractions": [],
            "restaurants": [],
            "accommodation": None,
            "transport_mode": None
        }
        
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan["attractions"].append({
                    "id": model.attr_data[a]['id'],
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration']
                })
        
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan["restaurants"].append({
                    "id": model.rest_data[r]['id'],
                    "name": model.rest_data[r]['name'],
                    "duration": model.rest_data[r]['duration']
                })
        
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    day_plan["accommodation"] = {
                        "id": model.hotel_data[h]['id'],
                        "name": model.hotel_data[h]['name']
                    }
        
        transport_mode = "taxi" if pyo.value(model.trans_mode[d]) < 0.5 else "bus"
        day_plan["transport_mode"] = transport_mode
        
        plan[f"day{d}"] = day_plan
    
    return plan

cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
solver = pyo.SolverFactory('scip')
results = solver.solve(model)

plan = generate_daily_plan(model, intra_city_trans)
print(f"```generated_plan
{plan}
```")