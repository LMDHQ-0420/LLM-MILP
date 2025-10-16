import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "苏州市"
destination_city = "武汉市"
budget = 35000
start_date = "2025年04月15日"
end_date = "2025年04月19日"
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
        'attractions': requests.get(url + f"/attractions/{destination_city}").json(),
        'accommodations': requests.get(url + f"/accommodations/{destination_city}").json(),
        'restaurants': requests.get(url + f"/restaurants/{destination_city}").json()
    }

    # 过滤景点和住宿
    poi_data['attractions'] = [a for a in poi_data['attractions'] if a['name'] in ['黄鹤楼', '归元寺', '长江大桥']]
    poi_data['accommodations'] = [h for h in poi_data['accommodations'] 
                                 if h['rating'] >= 4.7 and float(h['cost']) <= 1000]
    poi_data['restaurants'] = [r for r in poi_data['restaurants'] 
                               if any(food in r['recommended_food'] for food in ['武昌鱼', '鮰鱼'])]
    
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

    # 约束条件
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
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                       for r in model.restaurants)
        trans_time = sum(
            model.attr_hotel[d, a, h] * (
                model.trans_mode[d] * (get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                                     get_trans_params(intra_city_trans, a, h, 'bus_duration'))
            )
            for a in model.attractions for h in model.accommodations
        )
        return attr_time + rest_time + trans_time <= 840
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]
    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]
    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1
    model.link_attr_hotel1 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule1)
    model.link_attr_hotel2 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule2)
    model.link_attr_hotel3 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule3)

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 目标函数：最大化评分，尽可能花光预算
    def obj_rule(model):
        total_rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                          for d in model.days for a in model.attractions)
        total_rating += sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                          for d in model.days for r in model.restaurants)
        total_rating += sum(model.select_hotel[h] * model.hotel_data[h]['rating'] 
                          for h in model.accommodations) * 2
        
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
            for d in model.days for a in model.attractions for h in model.accommodations)
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                   for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                              for t in model.train_back)
        
        total_cost = ((peoples+1)//2 * hotel_cost + peoples * (attraction_cost + restaurant_cost + 
                     train_departure_cost + train_back_cost) + transport_cost)
        
        return -total_rating + (budget - total_cost)/10000
    
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    days = sorted(list(model.days))
    plan = {}
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取往返交通
    departure_train = next(t for t in model.train_departure if pyo.value(model.select_train_departure[t]) > 0.5)
    back_train = next(t for t in model.train_back if pyo.value(model.select_train_back[t]) > 0.5)
    
    plan['hotel'] = {
        'name': selected_hotel['name'],
        'cost_per_night': selected_hotel['cost'],
        'total_cost': selected_hotel['cost'] * (travel_days - 1)
    }
    
    plan['transport'] = {
        'departure': {
            'train_number': model.train_departure_data[departure_train]['train_number'],
            'cost': model.train_departure_data[departure_train]['cost'],
            'duration': model.train_departure_data[departure_train]['duration']
        },
        'return': {
            'train_number': model.train_back_data[back_train]['train_number'],
            'cost': model.train_back_data[back_train]['cost'],
            'duration': model.train_back_data[back_train]['duration']
        }
    }
    
    plan['daily_plan'] = {}
    for d in days:
        # 获取当天景点
        selected_attr = None
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                selected_attr = model.attr_data[a]
                break
        
        # 获取当天餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append(model.rest_data[r])
        
        # 获取交通方式
        transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "打车"
        
        plan['daily_plan'][d] = {
            'attraction': {
                'name': selected_attr['name'],
                'cost': selected_attr['cost'],
                'duration': selected_attr['duration']
            },
            'restaurants': [
                {
                    'name': r['name'],
                    'cost': r['cost'],
                    'duration': r['duration']
                } for r in selected_rests
            ],
            'transport_mode': transport_mode
        }
    
    return json.dumps(plan, indent=4, ensure_ascii=False)

# 主程序
cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
solver = pyo.SolverFactory('scip')
results = solver.solve(model, tee=True)

plan = generate_daily_plan(model, intra_city_trans)
print(f"```generated_plan
{plan}
```")