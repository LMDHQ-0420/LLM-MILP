import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "上海市"
destination_city = "贵阳市"
budget = 12000
start_date = "2025年05月28日"
end_date = "2025年05月30日"
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

    # 过滤景点只保留百花湖
    attractions = [a for a in poi_data['attractions'] if a['name'] == '百花湖']
    # 过滤酒店评分4.6以上
    hotels = [h for h in poi_data['accommodations'] if h['rating'] >= 4.6]
    # 过滤餐厅人均消费140元内且推荐铜锅牛肉
    restaurants = [r for r in poi_data['restaurants'] 
                  if float(r['cost']) <= 140 and '铜锅牛肉' in r['recommended_food']]

    attraction_dict = {a['id']: a for a in attractions}
    hotel_dict = {h['id']: h for h in hotels}
    restaurant_dict = {r['id']: r for r in restaurants}
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
            'destination_ststation': train_back_dict[t]['destination_station']
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

    # 约束条件：每天必须选一个景点
    def attr_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.attr_constr = pyo.Constraint(model.days, rule=attr_rule)

    # 约束条件：每天必须选3个餐厅
    def rest_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.rest_constr = pyo.Constraint(model.days, rule=rest_rule)

    # 约束条件：每天活动时间不超过840分钟
    def time_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
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

    model.time_constr = pyo.Constraint(model.days, rule=time_rule)

    # 约束条件：必须选择一个酒店
    def hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.hotel_constr = pyo.Constraint(rule=hotel_rule)

    # 约束条件：必须选择去程和返程火车
    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.train_departure_constr = pyo.Constraint(rule=train_departure_rule)
    model.train_back_constr = pyo.Constraint(rule=train_back_rule)

    # 约束条件：交通方式以公交为主
    def transport_rule(model, d):
        return model.trans_mode[d] == 1

    model.transport_constr = pyo.Constraint(model.days, rule=transport_rule)

    # 目标函数：最大化花费接近预算
    def obj_rule(model):
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
        total_cost = (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
                     attraction_cost + restaurant_cost + train_departure_cost + train_back_cost)
        return (total_cost - budget)**2  # 最小化与预算的差距

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'attractions': [],
            'restaurants': [],
            'hotel': None,
            'transport_mode': None
        }
        
        # 添加景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attractions'].append({
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration'],
                    'cost': model.attr_data[a]['cost']
                })
        
        # 添加餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'cost': model.rest_data[r]['cost'],
                    'recommended_food': model.rest_data[r]['recommended_food']
                })
        
        # 添加酒店（最后一天没有）
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    day_plan['hotel'] = {
                        'name': model.hotel_data[h]['name'],
                        'cost': model.hotel_data[h]['cost'],
                        'rating': model.hotel_data[h]['rating']
                    }
        
        # 添加交通方式
        day_plan['transport_mode'] = '公交' if pyo.value(model.trans_mode[d]) > 0.5 else '打车'
        
        plan[f'Day {d}'] = day_plan
    
    # 添加火车信息
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            plan['departure_train'] = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration']
            }
    
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            plan['return_train'] = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration']
            }
    
    return plan

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    plan = generate_daily_plan(model, intra_city_trans)
    print(f"```generated_plan\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")