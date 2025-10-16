import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "南京市"
destination_city = "厦门市"
budget = 5000
start_date = "2025年10月05日"
end_date = "2025年10月07日"
travel_days = 3
peoples = 1

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 获取景点数据并过滤评分高的热门景点
    attractions = requests.get(url + f"/attractions/{destination_city}").json()
    filtered_attractions = [a for a in attractions if a['name'] in ['鼓浪屿', '梦幻海岸']]

    # 获取酒店数据并过滤经济型连锁酒店
    accommodations = requests.get(url + f"/accommodations/{destination_city}").json()
    filtered_hotels = [h for h in accommodations if h['type'] == '经济型' and 
                      h['rating'] >= 4.5 and h['cost'] <= 800 and '连锁' in h['feature']]

    # 获取餐厅数据并过滤闽南菜
    restaurants = requests.get(url + f"/restaurants/{destination_city}").json()
    filtered_restaurants = [r for r in restaurants if '闽南菜' in r['type'] and 
                          '海蛎煎' in r['recommended_food']]

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
            'destination_station': train_back_dict[t['destination_station']]
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
        attraction_rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                               for d in model.days for a in model.attractions)
        restaurant_rating = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                              for d in model.days for r in model.restaurants)
        hotel_rating = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] 
                          for h in model.accommodations)
        
        restaurant_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] 
                            for d in model.days for r in model.restaurants)
        
        return - (attraction_rating + restaurant_rating + hotel_rating) + 0.1 * restaurant_time

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
        return hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    def daily_activity_time_rule(model, d):
        attraction_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                            for a in model.attractions)
        restaurant_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
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
        return attraction_time + restaurant_time + transport_time <= 840

    model.daily_activity_time = pyo.Constraint(model.days, rule=daily_activity_time_rule)

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

    def no_duplicate_attractions_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.no_duplicate_attractions = pyo.Constraint(model.attractions, rule=no_duplicate_attractions_rule)

    def no_duplicate_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.no_duplicate_restaurants = pyo.Constraint(model.restaurants, rule=no_duplicate_restaurants_rule)

    def public_transport_rule(model, d):
        return model.trans_mode[d] == 1

    model.public_transport = pyo.Constraint(model.days, rule=public_transport_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = {}
    date = datetime.strptime(start_date, "%Y年%m月%d日")
    
    # 获取选择的火车班次
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
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 生成每日计划
    for d in model.days:
        day_plan = {
            "date": date.strftime("%Y年%m月%d日"),
            "hotel": selected_hotel['name'] if d < travel_days else "无",
            "attractions": [],
            "restaurants": [],
            "transport_mode": "公交/地铁"
        }
        
        # 添加景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan["attractions"].append({
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration'],
                    "cost": model.attr_data[a]['cost']
                })
        
        # 添加餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan["restaurants"].append({
                    "name": model.rest_data[r]['name'],
                    "recommended_food": model.rest_data[r]['recommended_food'],
                    "duration": model.rest_data[r]['duration'],
                    "cost": model.rest_data[r]['cost']
                })
        
        plan[f"第{d}天"] = day_plan
        date += timedelta(days=1)
    
    # 添加交通信息
    plan["交通"] = {
        "去程": {
            "车次": selected_train_departure['train_number'],
            "出发站": selected_train_departure['origin_station'],
            "到达站": selected_train_departure['destination_station'],
            "出发时间": "上午",
            "时长": selected_train_departure['duration'],
            "费用": selected_train_departure['cost']
        },
        "返程": {
            "车次": selected_train_back['train_number'],
            "出发站": selected_train_back['origin_station'],
            "到达站": selected_train_back['destination_station'],
            "出发时间": "下午",
            "时长": selected_train_back['duration'],
            "费用": selected_train_back['cost']
        }
    }
    
    # 计算总费用
    total_cost = 0
    
    # 酒店费用
    hotel_cost = selected_hotel['cost'] * (travel_days - 1)
    total_cost += hotel_cost
    
    # 景点费用
    attraction_cost = 0
    for d in model.days:
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attraction_cost += model.attr_data[a]['cost']
    
    total_cost += attraction_cost
    
    # 餐厅费用
    restaurant_cost = 0
    for d in model.days:
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                restaurant_cost += model.rest_data[r]['cost']
    
    total_cost += restaurant_cost
    
    # 交通费用
    transport_cost = selected_train_departure['cost'] + selected_train_back['cost']
    total_cost += transport_cost
    
    plan["总费用"] = total_cost
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)

solver = pyo.SolverFactory('scip')
results = solver.solve(model, tee=True)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")
else:
    print("没有找到可行解")