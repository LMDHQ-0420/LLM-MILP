import math

import pyomo.environ as pyo
from datetime import datetime, timedelta
from pyomo.opt import TerminationCondition
import requests
from pyomo.opt import SolverStatus

# 用户输入
origin_city = "武汉市"
destination_city = "厦门市"
budget = 20000
start_date = "2025年7月1日"
end_date = "2025年7月3日"
travel_days = 3
peoples = 1


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
    hotel_dict = {h['id']: h for h in poi_data['accommodations']}
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure if t['train_number'] == 'D3286'}
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

    # 约束条件
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
        rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating']
                     for d in model.days for a in model.attractions) + \
                 sum(model.select_rest[d, r] * model.rest_data[r]['rating']
                     for d in model.days for r in model.restaurants) + \
                 sum(model.select_hotel[h] * model.hotel_data[h]['rating']
                     for h in model.accommodations)
        return rating

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                         for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                              for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                              for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.attr_hotel[d, a, h] * (
                    (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
            ) + \
                    peoples * model.trans_mode[d] * (
                            get_trans_params(intra_city_trans, h, a, 'bus_cost') + \
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

    model.budget_con = pyo.Constraint(rule=budget_rule)


    def time_rule(model, d):
        activity_time = sum(
            model.select_attr[d, a] * model.attr_data[a]['duration']
            for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
            for r in model.restaurants
        )
        trans_time = sum(
            model.attr_hotel[d, a, h] * (
                    (1 - model.trans_mode[d]) * (
                    get_trans_params(intra_city_trans, h, a, 'taxi_duration') + \
                    get_trans_params(intra_city_trans, a, h, 'taxi_duration')
            ) + \
                    model.trans_mode[d] * (
                            get_trans_params(intra_city_trans, h, a, 'bus_duration') + \
                            get_trans_params(intra_city_trans, a, h, 'bus_duration')
                    )
            )
            for a in model.attractions
            for h in model.accommodations
        )
        return activity_time + trans_time <= 840

    model.time_con = pyo.Constraint(model.days, rule=time_rule)

    model.unique_attr = pyo.Constraint(
        model.attractions,
        rule=lambda m, a: sum(m.select_attr[d, a] for d in m.days) <= 1
    )

    model.unique_rest = pyo.Constraint(
        model.restaurants,
        rule=lambda m, r: sum(m.select_rest[d, r] for d in m.days) <= 1
    )

    model.min_attr = pyo.Constraint(
        model.days,
        rule=lambda m, d: sum(m.select_attr[d, a] for a in m.attractions) == 1
    )

    model.min_rest = pyo.Constraint(
        model.days,
        rule=lambda m, d: sum(m.select_rest[d, r] for r in m.restaurants) == 3
    )

    model.single_train_departure = pyo.Constraint(
        rule=lambda m: sum(m.select_train_departure[t] for t in m.train_departure) == 1
    )

    model.single_train_back = pyo.Constraint(
        rule=lambda m: sum(m.select_train_back[t] for t in m.train_back) == 1
    )

    model.single_hotel = pyo.Constraint(
        rule=lambda m: sum(m.select_hotel[h] for h in m.accommodations) == 1
    )

    model.hotel_type = pyo.Constraint(
        rule=lambda m: sum(
            m.select_hotel[h] for h in m.accommodations
            if '三星级' in m.hotel_data[h]['type'] or '四星级' in m.hotel_data[h]['type'] or '五星级' in m.hotel_data[h]['type']
        ) == 1
    )


    model.hotel_type = pyo.Constraint(
        rule=lambda m: sum(
            m.select_hotel[h] for h in m.accommodations
            if '早餐' in m.hotel_data[h]['feature']
        ) == 1
    )

    model.feature_rest = pyo.Constraint(
        rule=lambda m: sum(
            model.select_rest[d, r]
            for r in model.restaurants
            for d in days
            if '茶艺馆' in model.rest_data[r]['type']
        ) >= 1
    )


    return model


# 生成日期范围
def generate_date_range(start_date, end_date, date_format="%Y年%m月%d日"):
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    days = travel_days
    return [
        (start + timedelta(days=i)).strftime(date_format)
        for i in range(days)
    ]


# 获取选择的列车
def get_selected_train(model, train_type='departure'):
    if train_type not in ['departure', 'back']:
        raise ValueError("train_type must in ['departure', 'back']")

    train_set = model.train_departure if train_type == 'departure' else model.train_back
    train_data = model.train_departure_data if train_type == 'departure' else model.train_back_data
    selected_train = [
        train_data[t]
        for t in train_set
        if pyo.value(
            model.select_train_departure[t] if train_type == 'departure'
            else model.select_train_back[t]
        ) > 0.9
    ]
    return selected_train[0]


# 获取选择的POI
def get_selected_poi(model, type, day, selected_poi, k=1):
    if type == 'restaurant':
        poi_set = model.restaurants
        poi_data = model.rest_data
        select_set = model.select_rest
    else:
        poi_set = model.attractions
        poi_data = model.attr_data
        select_set = model.select_attr

    selected_poi = [
        poi_data[t]
        for t in poi_set
        if t not in selected_poi and pyo.value(select_set[day, t]) > 0.9
    ]
    return selected_poi


# 获取选择的酒店
def get_selected_hotel(model):
    selected_hotel = [
        model.hotel_data[t]
        for t in model.accommodations
        if pyo.value(model.select_hotel[t]) > 0.9
    ]
    return selected_hotel[0]


# 计算时间
def get_time(model, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day, intra_city_trans):
    daily_time = 0
    daily_time += selected_attr['duration']
    for r in selected_rest:
        daily_time += r['queue_time'] + r['duration']

    if pyo.value(model.trans_mode[day]) > 0.9:
        transport_time = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'bus_duration'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'bus_duration'
        )
    else:
        transport_time = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'taxi_duration'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'taxi_duration'
        )

    return daily_time + transport_time, transport_time


# 计算费用
def get_cost(model, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day, intra_city_trans):
    daily_cost = 0
    daily_cost += peoples * selected_attr['cost']
    for r in selected_rest:
        daily_cost += peoples * r['cost']

    if pyo.value(model.trans_mode[day]) > 0.9:
        transport_cost = peoples * get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'bus_cost'
        ) + peoples * get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'bus_cost'
        )
    else:
        transport_cost = get_trans_params(
            intra_city_trans,
            selected_hotel['id'],
            selected_attr['id'],
            'taxi_cost'
        ) + get_trans_params(
            intra_city_trans,
            selected_attr['id'],
            selected_hotel['id'],
            'taxi_cost'
        )

    if day != travel_days:
        daily_cost += selected_hotel['cost']
    if day == 1:
        daily_cost += peoples * departure_trains['cost']
    if day == travel_days:
        daily_cost += peoples * back_trains['cost']
    return daily_cost + transport_cost, transport_cost

def generate_poi(model, intra_city_trans):
    departure_trains = get_selected_train(model, 'departure')
    back_trains = get_selected_train(model, 'back')
    selected_hotel = get_selected_hotel(model)['id']

    select_at = []
    select_re = []
    for day in sorted(model.days):
        attr_details = []
        rest_details = []
        attr_details = get_selected_poi(model, 'attraction', day, select_at)[0]
        select_at.append(attr_details['id'])
        rest_details = get_selected_poi(model, 'restaurant', day, select_re)
        for r in rest_details:
            select_re.append(r['id'])

    return {
        "query_id": "2",
        "query": "2025年7月1日上午，我将从武汉大学地铁站启程前往厦门，开启为期3天的旅行，并于7月3日晚搭乘高铁返程。出发交通指定搭乘D3286次高铁，本次旅行预算控制在20000元以内，期望能深入体验厦门的本地文化与自然风景。请优先安排POI评分高的景点、餐厅与住宿，入住三星级以上宾馆，要求含早餐，且至少吃一次茶艺馆，并确保每日动线紧凑、交通方便，提升整体出行舒适度。",
        "travel_days":travel_days,
        "budget": budget,
        "peoples": peoples,
        "origin_city": origin_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "departure_trains": {
            'train_number': departure_trains['train_number'],
            'origin_id': departure_trains['origin_id'],
            'destination_id': departure_trains['destination_id']
        },
        "back_trains": {
            'train_number': back_trains['train_number'],
            'origin_id': back_trains['origin_id'],
            'destination_id': back_trains['destination_id']
        },
        "hotels": selected_hotel,
        "attractions": select_at,
        "restaurants": select_re
    }

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    departure_trains = get_selected_train(model, 'departure')
    back_trains = get_selected_train(model, 'back')
    selected_hotel = get_selected_hotel(model)
    total_cost = 0
    daily_plans = []
    select_at = []
    select_re = []
    date = generate_date_range(start_date, end_date)
    for day in sorted(model.days):
        attr_details = get_selected_poi(model, 'attraction', day, select_at)[0]
        select_at.append(attr_details['id'])
        rest_details = get_selected_poi(model, 'restaurant', day, select_re)
        for r in rest_details:
            select_re.append(r['id'])
        meal_allocation = {
            'breakfast': rest_details[0],
            'lunch': rest_details[1],
            'dinner': rest_details[2]
        }

        daily_time, transport_time = get_time(model, attr_details, rest_details, departure_trains, back_trains,
                                              selected_hotel, day, intra_city_trans)
        daily_cost, transport_cost = get_cost(model, attr_details, rest_details, departure_trains, back_trains,
                                              selected_hotel, day, intra_city_trans)
        day_plan = {
            "date": f"{date[day - 1]}",
            "cost": round(daily_cost, 2),
            "cost_time": round(daily_time, 2),
            "hotel": selected_hotel if day != travel_days else "null",
            "attractions": attr_details,
            "restaurants": [
                {
                    "type": meal_type,
                    "restaurant": rest if rest else None
                } for meal_type, rest in meal_allocation.items()
            ],
            "transport": {
                "mode": "bus" if pyo.value(model.trans_mode[day]) > 0.9 else "taxi",
                "cost": round(transport_cost, 2),
                "duration": round(transport_time, 2)
            }
        }
        daily_plans.append(day_plan)
        total_cost += daily_cost

    return {
        "budget": budget,
        "peoples": peoples,
        "travel_days": travel_days,
        "origin_city": origin_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "daily_plans": daily_plans,
        "departure_trains": departure_trains,
        "back_trains": back_trains,
        "total_cost": round(total_cost, 2),
        "objective_value": round(pyo.value(model.obj), 2)
    }


# 配置求解器
def configure_solver():
    solver = pyo.SolverFactory('scip')
    solver.options = {
        'limits/time': 300,
        'limits/gap': 0,
    }
    return solver


# 求解旅行计划
def solve_travel_plan(data):
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = data
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    solver = configure_solver()
    results = solver.solve(model, tee=True)
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{plan}\n```")


if __name__ == "__main__":
    data = fetch_data()
    solve_travel_plan(data)
