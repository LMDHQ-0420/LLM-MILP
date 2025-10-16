import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "杭州市"
destination_city = "北京市"
budget = 13000
start_date = "2025年7月1日"
end_date = "2025年7月5日"
travel_days = 5
peoples = 2

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    attractions = requests.get(url + f"/attractions/{destination_city}").json()
    # 筛选指定景点
    filtered_attractions = [a for a in attractions if a['name'] in ['故宫', '颐和园']]
    
    accommodations = requests.get(url + f"/accommodations/{destination_city}").json()
    # 筛选主题酒店且评分4.7以上
    filtered_accommodations = [h for h in accommodations if 
                              h['type'] == '主题酒店' and 
                              float(h['rating']) >= 4.7 and 
                              '含早餐' in h['feature']]
    
    restaurants = requests.get(url + f"/restaurants/{destination_city}").json()
    # 筛选推荐北京烤鸭且人均消费160元内
    filtered_restaurants = [r for r in restaurants if 
                           '北京烤鸭' in r['recommended_food'] and 
                           float(r['cost']) <= 160]

    poi_data = {
        'attractions': filtered_attractions,
        'accommodations': filtered_accommodations,
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
            'queue_time': float(restaurant_dict[r]['queue_time']),
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 默认公交
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

    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.one_attr_per_day = pyo.Constraint(model.days, rule=one_attr_per_day_rule)

    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.three_rest_per_day = pyo.Constraint(model.days, rule=three_rest_per_day_rule)

    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
        trans_time = 0
        if d < travel_days:
            for a in model.attractions:
                for h in model.accommodations:
                    trans_time += model.attr_hotel[d, a, h] * (
                        2 * (model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                            (1-model.trans_mode[d]) * get_trans_params(intra_city_trans, h, a, 'taxi_duration'))
                    )
        return attr_time + rest_time + trans_time <= 840
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                (1-model.trans_mode[d]) * (get_trans_params(intra_city_trans, h, a, 'taxi_cost') + get_trans_params(intra_city_trans, a, h, 'taxi_cost')) +
                peoples * model.trans_mode[d] * (get_trans_params(intra_city_trans, h, a, 'bus_cost') + get_trans_params(intra_city_trans, a, h, 'bus_cost'))
            )
            for d in model.days for a in model.attractions for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        return (peoples+1)//2 * hotel_cost + peoples * (attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) + transport_cost <= budget
    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    def obj_rule(model):
        rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions) + \
                sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants) + \
                sum(model.select_hotel[h] * model.hotel_data[h]['rating'] * (travel_days - 1) for h in model.accommodations)
        queue_time = sum(model.select_rest[d, r] * model.rest_data[r]['queue_time'] for d in model.days for r in model.restaurants)
        return rating * 100 - queue_time
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            "attraction": None,
            "restaurants": [],
            "hotel": None,
            "transport_mode": None
        }
        
        # 景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan["attraction"] = {
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration'],
                    "cost": model.attr_data[a]['cost']
                }
                break
        
        # 餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan["restaurants"].append({
                    "name": model.rest_data[r]['name'],
                    "duration": model.rest_data[r]['duration'],
                    "queue_time": model.rest_data[r]['queue_time'],
                    "cost": model.rest_data[r]['cost']
                })
        
        # 酒店
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    day_plan["hotel"] = {
                        "name": model.hotel_data[h]['name'],
                        "cost": model.hotel_data[h]['cost']
                    }
                    break
        
        # 交通方式
        transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "打车"
        day_plan["transport_mode"] = transport_mode
        
        plan[f"Day {d}"] = day_plan
    
    # 往返火车
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = {
                "train_number": model.train_departure_data[t]['train_number'],
                "duration": model.train_departure_data[t]['duration'],
                "cost": model.train_departure_data[t]['cost']
            }
            break
    
    back_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            back_train = {
                "train_number": model.train_back_data[t]['train_number'],
                "duration": model.train_back_data[t]['duration'],
                "cost": model.train_back_data[t]['cost']
            }
            break
    
    plan["Transportation"] = {
        "Departure": departure_train,
        "Return": back_train
    }
    
    return json.dumps(plan, indent=2, ensure_ascii=False)

# 主程序
cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
solver = pyo.SolverFactory('scip')
results = solver.solve(model, tee=True)

plan = generate_daily_plan(model, intra_city_trans)
print(f"```generated_plan
{plan}
```")