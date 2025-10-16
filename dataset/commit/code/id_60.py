import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "广州市"
destination_city = "洛阳市"
budget = 11000
start_date = "2025年9月22日"
end_date = "2025年9月24日"
travel_days = 3
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

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations'] if float(h['rating']) >= 4.7}
    restaurant_dict = {r['id']: r for r in poi_data['restaurants'] if any('牡丹燕菜' in food for food in r.get('recommended_food', []))}
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}

    # 确保必选景点存在
    required_attractions = ['龙门石窟', '洛邑古城']
    for name in required_attractions:
        if not any(a['name'] == name for a in poi_data['attractions']):
            raise ValueError(f"必选景点 {name} 不存在")

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

    # 必选景点约束
    def required_attractions_rule(model, d):
        required_attrs = [a for a in model.attractions if model.attr_data[a]['name'] in ['龙门石窟', '洛邑古城']]
        return sum(model.select_attr[d, a] for a in required_attrs) >= 1

    model.required_attractions = pyo.Constraint(model.days, rule=required_attractions_rule)

    # 每日活动时间约束
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

    # 每日景点约束
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)

    # 每日餐厅约束
    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    # 餐厅不重复约束
    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)

    # 景点不重复约束
    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)

    # 酒店选择约束
    def hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.hotel_selection = pyo.Constraint(rule=hotel_rule)

    # 火车选择约束
    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.train_departure_selection = pyo.Constraint(rule=train_departure_rule)
    model.train_back_selection = pyo.Constraint(rule=train_back_rule)

    # 预算约束
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
                    get_trans_params(intra_city_trans, h, a, 'taxi_cost') + 
                    get_trans_params(intra_city_trans, a, h, 'taxi_cost')
                ) + 
                model.trans_mode[d] * (
                    get_trans_params(intra_city_trans, h, a, 'bus_cost') + 
                    get_trans_params(intra_city_trans, a, h, 'bus_cost')
                )
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                   for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                              for t in model.train_back)
        return (peoples+1)//2 * hotel_cost + transport_cost + peoples * (
            attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    # 目标函数：最大化景点时长，最小化就餐等待
    def obj_rule(model):
        attraction_duration = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                                 for d in model.days for a in model.attractions)
        restaurant_waiting = sum(model.select_rest[d, r] * model.rest_data[r]['queue_time'] 
                                for d in model.days for r in model.restaurants)
        return -attraction_duration + restaurant_waiting

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans, train_departure_data, train_back_data):
    plan = {}
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'attractions': [],
            'restaurants': [],
            'hotel': None,
            'transport_mode': None,
            'train': None
        }

        # 景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attr_data = model.attr_data[a]
                day_plan['attractions'].append({
                    'name': attr_data['name'],
                    'duration': attr_data['duration'],
                    'cost': attr_data['cost']
                })

        # 餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                rest_data = model.rest_data[r]
                day_plan['restaurants'].append({
                    'name': rest_data['name'],
                    'duration': rest_data['duration'],
                    'queue_time': rest_data['queue_time'],
                    'cost': rest_data['cost'],
                    'recommended_food': rest_data['recommended_food']
                })

        # 酒店
        for h in model.accommodations:
            if pyo.value(model.select_hotel[h]) > 0.5:
                if d < travel_days:  # 最后一天不安排酒店
                    hotel_data = model.hotel_data[h]
                    day_plan['hotel'] = {
                        'name': hotel_data['name'],
                        'cost': hotel_data['cost'],
                        'rating': hotel_data['rating']
                    }

        # 交通方式
        transport_mode = "taxi" if pyo.value(model.trans_mode[d]) < 0.5 else "bus"
        day_plan['transport_mode'] = transport_mode

        # 火车
        if d == 1:
            for t in model.train_departure:
                if pyo.value(model.select_train_departure[t]) > 0.5:
                    train_data = train_departure_data[t]
                    day_plan['train'] = {
                        'train_number': train_data['train_number'],
                        'departure_station': train_data['origin_station'],
                        'arrival_station': train_data['destination_station'],
                        'duration': train_data['duration'],
                        'cost': train_data['cost']
                    }
        elif d == travel_days:
            for t in model.train_back:
                if pyo.value(model.select_train_back[t]) > 0.5:
                    train_data = train_back_data[t]
                    day_plan['train'] = {
                        'train_number': train_data['train_number'],
                        'departure_station': train_data['origin_station'],
                        'arrival_station': train_data['destination_station'],
                        'duration': train_data['duration'],
                        'cost': train_data['cost']
                    }

        plan[f"Day {d}"] = day_plan

    # 计算总花费
    total_cost = 0
    # 酒店费用
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            total_cost += (peoples+1)//2 * pyo.value(model.hotel_data[h]['cost']) * (travel_days - 1)
    
    # 景点费用
    for d in model.days:
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                total_cost += peoples * pyo.value(model.attr_data[a]['cost'])
    
    # 餐厅费用
    for d in model.days:
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                total_cost += peoples * pyo.value(model.rest_data[r]['cost'])
    
    # 市内交通费用
    transport_cost = 0
    for d in model.days:
        for a in model.attractions:
            for h in model.accommodations:
                if pyo.value(model.attr_hotel[d, a, h]) > 0.5:
                    if pyo.value(model.trans_mode[d]) < 0.5:  # taxi
                        transport_cost += pyo.value(get_trans_params(intra_city_trans, h, a, 'taxi_cost'))
                        transport_cost += pyo.value(get_trans_params(intra_city_trans, a, h, 'taxi_cost'))
                    else:  # bus
                        transport_cost += peoples * pyo.value(get_trans_params(intra_city_trans, h, a, 'bus_cost'))
                        transport_cost += peoples * pyo.value(get_trans_params(intra_city_trans, a, h, 'bus_cost'))
    total_cost += transport_cost
    
    # 火车费用
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            total_cost += peoples * pyo.value(model.train_departure_data[t]['cost'])
    
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            total_cost += peoples * pyo.value(model.train_back_data[t]['cost'])

    plan['total_cost'] = total_cost
    return plan

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans, cross_city_train_departure, cross_city_train_back)
        print(f"```generated_plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```")
    else:
        print("No solution found")

if __name__ == "__main__":
    main()