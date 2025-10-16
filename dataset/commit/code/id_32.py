import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "武汉市"
destination_city = "三亚市"
budget = 0  # 不限制预算
start_date = "2025年03月20日"
end_date = "2025年03月24日"
travel_days = 5
peoples = 3

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

# 获取交通参数
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

# 构建模型
def build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans):
    model = pyo.ConcreteModel()

    # 定义集合
    days = list(range(1, travel_days + 1))
    model.days = pyo.Set(initialize=days)

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations'] if float(h['cost']) <= 1500}  # 精品酒店限制
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}
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
            'destination_id': train_back_dict[t['destination_id']],
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

    # 每日必须选择一个景点
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)

    # 景点不重复
    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)

    # 每日选择3个餐厅
    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    # 餐厅不重复
    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)

    # 只选择一个酒店
    def single_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.single_hotel = pyo.Constraint(rule=single_hotel_rule)

    # 只选择一个出发车次和一个返回车次
    def single_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def single_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.single_train_departure = pyo.Constraint(rule=single_train_departure_rule)
    model.single_train_back = pyo.Constraint(rule=single_train_back_rule)

    # 每日活动时间不超过840分钟
    def daily_time_rule(model, d):
        attraction_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        restaurant_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
        
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
        
        return attraction_time + restaurant_time + transport_time <= 840

    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    # 目标函数：最大化评分
    def obj_rule(model):
        attraction_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                              for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] 
                          for h in model.accommodations)
        restaurant_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                              for d in model.days for r in model.restaurants)
        return -(attraction_score + hotel_score + restaurant_score)  # 最小化负分=最大化正分

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = []
    date = datetime.strptime(start_date, "%Y年%m月%d日")
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = h
            break
    
    # 获取交通方式
    transport_modes = {}
    for d in model.days:
        transport_modes[d] = "taxi" if pyo.value(model.trans_mode[d]) < 0.5 else "bus"
    
    # 获取出发和返回车次
    departure_train = next(t for t in model.train_departure if pyo.value(model.select_train_departure[t]) > 0.5)
    back_train = next(t for t in model.train_back if pyo.value(model.select_train_back[t]) > 0.5)
    
    # 第一天
    day1_plan = {
        "date": date.strftime("%Y年%m月%d日"),
        "description": "出发日",
        "transport": {
            "type": "train",
            "train_number": model.train_departure_data[departure_train]['train_number'],
            "departure_station": model.train_departure_data[departure_train]['origin_station'],
            "arrival_station": model.train_departure_data[departure_train]['destination_station'],
            "departure_time": "上午",
            "duration": model.train_departure_data[departure_train]['duration'],
            "cost": model.train_departure_data[departure_train]['cost']
        },
        "hotel": None,
        "attractions": [],
        "restaurants": []
    }
    
    # 获取第一天的景点和餐厅
    for a in model.attractions:
        if pyo.value(model.select_attr[1, a]) > 0.5:
            day1_plan["attractions"].append({
                "name": model.attr_data[a]['name'],
                "duration": model.attr_data[a]['duration'],
                "cost": model.attr_data[a]['cost']
            })
    
    for r in model.restaurants:
        if pyo.value(model.select_rest[1, r]) > 0.5:
            day1_plan["restaurants"].append({
                "name": model.rest_data[r]['name'],
                "duration": model.rest_data[r]['duration'],
                "cost": model.rest_data[r]['cost'],
                "recommended_food": model.rest_data[r]['recommended_food']
            })
    
    plan.append(day1_plan)
    date += timedelta(days=1)
    
    # 中间几天
    for d in range(2, travel_days):
        daily_plan = {
            "date": date.strftime("%Y年%m月%d日"),
            "description": f"第{d}天",
            "transport": {
                "type": "intra-city",
                "mode": transport_modes[d],
                "cost": sum(
                    pyo.value(model.attr_hotel[d, a, selected_hotel]) * (
                        (1 - pyo.value(model.trans_mode[d])) * (
                            get_trans_params(intra_city_trans, selected_hotel, a, 'taxi_cost') + 
                            get_trans_params(intra_city_trans, a, selected_hotel, 'taxi_cost')
                        ) + 
                        peoples * pyo.value(model.trans_mode[d]) * (
                            get_trans_params(intra_city_trans, selected_hotel, a, 'bus_cost') + 
                            get_trans_params(intra_city_trans, a, selected_hotel, 'bus_cost')
                        )
                    )
                    for a in model.attractions
                ),
                "duration": sum(
                    pyo.value(model.attr_hotel[d, a, selected_hotel]) * (
                        (1 - pyo.value(model.trans_mode[d])) * (
                            get_trans_params(intra_city_trans, selected_hotel, a, 'taxi_duration') + 
                            get_trans_params(intra_city_trans, a, selected_hotel, 'taxi_duration')
                        ) + 
                        pyo.value(model.trans_mode[d]) * (
                            get_trans_params(intra_city_trans, selected_hotel, a, 'bus_duration') + 
                            get_trans_params(intra_city_trans, a, selected_hotel, 'bus_duration')
                        )
                    )
                    for a in model.attractions
                )
            },
            "hotel": {
                "name": model.hotel_data[selected_hotel]['name'],
                "cost": model.hotel_data[selected_hotel]['cost'],
                "feature": model.hotel_data[selected_hotel]['feature']
            },
            "attractions": [],
            "restaurants": []
        }
        
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                daily_plan["attractions"].append({
                    "name": model.attr_data[a]['name'],
                    "duration": model.attr_data[a]['duration'],
                    "cost": model.attr_data[a]['cost']
                })
        
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                daily_plan["restaurants"].append({
                    "name": model.rest_data[r]['name'],
                    "duration": model.rest_data[r]['duration'],
                    "cost": model.rest_data[r]['cost'],
                    "recommended_food": model.rest_data[r]['recommended_food']
                })
        
        plan.append(daily_plan)
        date += timedelta(days=1)
    
    # 最后一天
    last_day_plan = {
        "date": date.strftime("%Y年%m月%d日"),
        "description": "返程日",
        "transport": {
            "type": "train",
            "train_number": model.train_back_data[back_train]['train_number'],
            "departure_station": model.train_back_data[back_train]['origin_station'],
            "arrival_station": model.train_back_data[back_train]['destination_station'],
            "departure_time": "下午",
            "duration": model.train_back_data[back_train]['duration'],
            "cost": model.train_back_data[back_train]['cost']
        },
        "hotel": None,
        "attractions": [],
        "restaurants": []
    }
    
    for a in model.attractions:
        if pyo.value(model.select_attr[travel_days, a]) > 0.5:
            last_day_plan["attractions"].append({
                "name": model.attr_data[a]['name'],
                "duration": model.attr_data[a]['duration'],
                "cost": model.attr_data[a]['cost']
            })
    
    for r in model.restaurants:
        if pyo.value(model.select_rest[travel_days, r]) > 0.5:
            last_day_plan["restaurants"].append({
                "name": model.rest_data[r]['name'],
                "duration": model.rest_data[r]['duration'],
                "cost": model.rest_data[r]['cost'],
                "recommended_food": model.rest_data[r]['recommended_food']
            })
    
    plan.append(last_day_plan)
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")