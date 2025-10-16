import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "武汉市"
destination_city = "苏州市"
budget = 0  # 不限预算
start_date = "2025年09月01日"
end_date = "2025年09月05日"
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

    # 过滤景点：只保留平江路、留园、金鸡湖等经典景点
    filtered_attractions = [a for a in poi_data['attractions'] if a['name'] in ['平江路', '留园', '金鸡湖']]
    attraction_dict = {a['id']: a for a in filtered_attractions}
    
    # 过滤酒店：评分高、含早餐、价格<=1200元
    filtered_hotels = [h for h in poi_data['accommodations'] 
                      if h['rating'] >= 4.5 and '早餐' in h['feature'] and float(h['cost']) <= 1200]
    hotel_dict = {h['id']: h for h in filtered_hotels}
    
    # 过滤餐厅：江苏菜或精致茶点，评分高，排队时间短
    filtered_restaurants = [r for r in poi_data['restaurants'] 
                           if ('江苏菜' in r['type'] or '茶点' in r['type']) 
                           and r['rating'] >= 4.5 and float(r['queue_time']) <= 30]
    restaurant_dict = {r['id']: r for r in filtered_restaurants}
    
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

    # 约束条件：每天选择一个景点
    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.one_attr_per_day = pyo.Constraint(model.days, rule=one_attr_per_day_rule)

    # 约束条件：景点不重复
    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.unique_attr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 约束条件：每天选择3个餐厅
    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.three_rest_per_day = pyo.Constraint(model.days, rule=three_rest_per_day_rule)

    # 约束条件：餐厅不重复
    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.unique_rest = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 约束条件：选择一个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 约束条件：每天活动时间不超过840分钟
    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
                        for r in model.restaurants)
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
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    # 约束条件：选择一个去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    # 约束条件：选择一个返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 目标函数：最大化评分
    def obj_rule(model):
        hotel_rating = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations)
        attraction_rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                               for d in model.days for a in model.attractions)
        restaurant_rating = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                              for d in model.days for r in model.restaurants)
        return -(hotel_rating + attraction_rating + restaurant_rating)  # 负号因为是最小化

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    date = datetime.strptime(start_date, "%Y年%m月%d日")
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取去程火车
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = model.train_departure_data[t]
            break
    
    # 获取返程火车
    back_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            back_train = model.train_back_data[t]
            break
    
    plan['summary'] = {
        'hotel': selected_hotel['name'],
        'departure_train': f"{departure_train['train_number']} ({departure_train['origin_station']} -> {departure_train['destination_station']})",
        'return_train': f"{back_train['train_number']} ({back_train['origin_station']} -> {back_train['destination_station']})",
        'total_days': travel_days
    }
    
    plan['days'] = []
    for d in model.days:
        day_plan = {
            'date': date.strftime("%Y年%m月%d日"),
            'attractions': [],
            'restaurants': [],
            'transport_mode': [],
            'hotel': selected_hotel['name'] if d < travel_days else None
        }
        
        # 获取景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attractions'].append({
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration'],
                    'rating': model.attr_data[a]['rating']
                })
                break
        
        # 获取餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'type': model.rest_data[r]['type'],
                    'recommended_food': model.rest_data[r]['recommended_food'],
                    'rating': model.rest_data[r]['rating']
                })
        
        # 获取交通方式
        transport_mode = "打车" if pyo.value(model.trans_mode[d]) < 0.5 else "公交"
        day_plan['transport_mode'] = transport_mode
        
        plan['days'].append(day_plan)
        date += timedelta(days=1)
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")