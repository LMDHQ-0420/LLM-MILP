import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "广州市"
destination_city = "厦门市"
budget = 12000
start_date = "2025年09月10日"
end_date = "2025年09月14日"
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
    days = list(range(1, travel_days))
    model.days = pyo.Set(initialize=days)

    # 过滤景点
    required_attractions = ['鼓浪屿', '南普陀寺']
    attractions = [a for a in poi_data['attractions'] if a['name'] in required_attractions]
    attraction_dict = {a['id']: a for a in attractions}
    
    # 过滤酒店
    hotels = [h for h in poi_data['accommodations'] if h['rating'] >= 4.8 and 'breakfast' in h['feature'].lower()]
    hotel_dict = {h['id']: h for h in hotels}
    
    # 过滤餐厅
    restaurants = [r for r in poi_data['restaurants'] if '沙茶面' in r['recommended_food'] and r['cost'] <= 180]
    restaurant_dict = {r['id']: r for r in restaurants}
    
    # 过滤火车班次
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}
    
    # 找到运行时间最短的火车班次
    min_departure_duration = min(t['duration'] for t in cross_city_train_departure)
    min_back_duration = min(t['duration'] for t in cross_city_train_back)
    
    model.attractions = pyo.Set(initialize=attraction_dict.keys())
    model.accommodations = pyo.Set(initialize=hotel_dict.keys())
    model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
    model.train_departure = pyo.Set(initialize=[t['train_number'] for t in cross_city_train_departure if t['duration'] == min_departure_duration])
    model.train_back = pyo.Set(initialize=[t['train_number'] for t in cross_city_train_back if t['duration'] == min_back_duration])

    # 定义参数
    model.attr_data = pyo.Param(
        model.attractions,
        initialize=lambda m, a: {
            'id': attraction_dict[a]['id'],
            'name': attraction_dict[a]['name'],
            'cost': float(attraction_dict[a]['cost']),
            'duration': float(attraction_dict[a]['duration']),
            'rating': float(attraction_dict[a]['rating'])
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
            'rating': float(restaurant_dict[r]['rating'])
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

    # 定义变量
    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)

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

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants)
        trans_time = sum(
            (1 - model.trans_mode[d]) * get_trans_params(intra_city_trans, h, a, 'taxi_duration') * 2 +
            model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_duration') * 2
            for a in model.attractions
            for h in model.accommodations
            if model.select_attr[d, a].value == 1 and model.select_hotel[h].value == 1
        )
        return attr_time + rest_time + trans_time <= 840
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants)
        transport_cost = sum(
            (1 - model.trans_mode[d]) * get_trans_params(intra_city_trans, h, a, 'taxi_cost') * 2 +
            model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_cost') * 2 * peoples
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
            if model.select_attr[d, a].value == 1 and model.select_hotel[h].value == 1
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        return (hotel_cost + transport_cost + peoples * (attraction_cost + restaurant_cost + train_departure_cost + train_back_cost)) <= budget
    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # 目标函数：最大化评分
    def obj_rule(model):
        attr_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations)
        rest_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants)
        return -(attr_score + hotel_score + rest_score)  # 最小化负分数即最大化分数
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = []
    for d in model.days:
        day_plan = {
            'day': d,
            'attraction': None,
            'restaurants': [],
            'hotel': None,
            'transport_mode': None
        }
        
        # 景点
        for a in model.attractions:
            if model.select_attr[d, a].value == 1:
                day_plan['attraction'] = {
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration']
                }
                break
                
        # 餐厅
        for r in model.restaurants:
            if model.select_rest[d, r].value == 1:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'duration': model.rest_data[r]['duration']
                })
                
        # 酒店
        if d < travel_days - 1:  # 最后一天不选酒店
            for h in model.accommodations:
                if model.select_hotel[h].value == 1:
                    day_plan['hotel'] = {
                        'name': model.hotel_data[h]['name']
                    }
                    break
                    
        # 交通方式
        day_plan['transport_mode'] = 'taxi' if model.trans_mode[d].value == 0 else 'bus'
        
        plan.append(day_plan)
    
    # 添加火车信息
    for t in model.train_departure:
        if model.select_train_departure[t].value == 1:
            plan.insert(0, {
                'transport': {
                    'type': 'train',
                    'number': model.train_departure_data[t]['train_number'],
                    'direction': 'departure'
                }
            })
    
    for t in model.train_back:
        if model.select_train_back[t].value == 1:
            plan.append({
                'transport': {
                    'type': 'train',
                    'number': model.train_back_data[t]['train_number'],
                    'direction': 'return'
                }
            })
    
    return plan

# 主程序
cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
solver = pyo.SolverFactory('scip')
results = solver.solve(model)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```")
else:
    print("No optimal solution found")