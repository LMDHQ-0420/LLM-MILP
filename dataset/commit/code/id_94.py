import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "广州市"
destination_city = "三亚市"
budget = 47000
start_date = "2025年08月18日"
end_date = "2025年08月23日"
travel_days = 6
peoples = 4

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 获取POI数据并过滤
    attractions = [a for a in requests.get(url + f"/attractions/{destination_city}").json() 
                  if a['name'] in ["大小洞天", "红色娘子军景区"]]
    accommodations = [h for h in requests.get(url + f"/accommodations/{destination_city}").json() 
                    if float(h['rating']) >= 4.8 and '全餐' in h['feature']]
    restaurants = [r for r in requests.get(url + f"/restaurants/{destination_city}").json() 
                 if '海南鸡饭' in r['recommended_food'] and float(r['cost']) <= 220]

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
            'duration': float(restaurant_dict[r]['duration']),
            'queue_time': float(restaurant_dict[r]['queue_time'])
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
    model.attr_hotel = pyo.Var(model.days, model.attractions, model.accommodations, domain=pyo.Binary)

    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]

    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]

    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

    model.link_attr_hotel1 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule1)
    model.link_attr_hotel2 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule2)
    model.link_attr_hotel3 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule3)

    # 约束条件
    def one_attraction_per_day(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    def three_restaurants_per_day(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    def one_hotel(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    def unique_attractions(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    def unique_restaurants(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    def time_constraint(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
        trans_time = sum(
            model.attr_hotel[d, a, h] * (
                (1 - model.trans_mode[d]) * (get_trans_params(intra_city_trans, h, a, 'taxi_duration') + 
                                           get_trans_params(intra_city_trans, a, h, 'taxi_duration')) +
                model.trans_mode[d] * (get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                                      get_trans_params(intra_city_trans, a, h, 'bus_duration'))
            )
            for a in model.attractions for h in model.accommodations
        )
        return attr_time + rest_time + trans_time <= 840

    def one_train_departure(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def one_train_back(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day)
    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day)
    model.one_hotel = pyo.Constraint(rule=one_hotel)
    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions)
    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants)
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint)
    model.one_train_departure = pyo.Constraint(rule=one_train_departure)
    model.one_train_back = pyo.Constraint(rule=one_train_back)

    # 预算约束
    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                (1 - model.trans_mode[d]) * (get_trans_params(intra_city_trans, h, a, 'taxi_cost') + 
                                           get_trans_params(intra_city_trans, a, h, 'taxi_cost')) +
                peoples * model.trans_mode[d] * (get_trans_params(intra_city_trans, h, a, 'bus_cost') + 
                                               get_trans_params(intra_city_trans, a, h, 'bus_cost'))
            )
            for d in model.days for a in model.attractions for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        return (peoples+1)//2 * hotel_cost + transport_cost + peoples * (attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # 目标函数：最大化游玩时间
    def obj_rule(model):
        return sum(
            model.select_attr[d, a] * model.attr_data[a]['duration'] +
            sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants)
            for d in model.days for a in model.attractions
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    selected_hotel = next(h for h in model.accommodations if model.select_hotel[h].value == 1)
    hotel_name = model.hotel_data[selected_hotel]['name']
    
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'hotel': hotel_name if d < travel_days else "无",
            'attraction': {},
            'restaurants': [],
            'transport_mode': "打车" if model.trans_mode[d].value == 0 else "公交"
        }
        
        selected_attr = next(a for a in model.attractions if model.select_attr[d, a].value == 1)
        day_plan['attraction'] = {
            'name': model.attr_data[selected_attr]['name'],
            'duration': model.attr_data[selected_attr]['duration']
        }
        
        for r in model.restaurants:
            if model.select_rest[d, r].value == 1:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'duration': model.rest_data[r]['duration']
                })
        
        plan[f"Day {d}"] = day_plan
    
    selected_train_departure = next(t for t in model.train_departure if model.select_train_departure[t].value == 1)
    selected_train_back = next(t for t in model.train_back if model.select_train_back[t].value == 1)
    
    plan['transport'] = {
        'departure': {
            'train_number': model.train_departure_data[selected_train_departure]['train_number'],
            'duration': model.train_departure_data[selected_train_departure]['duration']
        },
        'return': {
            'train_number': model.train_back_data[selected_train_back]['train_number'],
            'duration': model.train_back_data[selected_train_back]['duration']
        }
    }
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")