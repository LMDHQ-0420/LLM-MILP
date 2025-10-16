import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "重庆市"
destination_city = "武汉市"
budget = 22000
start_date = "2025年09月15日"
end_date = "2025年09月19日"
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

    # 过滤景点（必须包含武汉防汛纪念碑）
    attractions = [a for a in poi_data['attractions']]
    flood_monument = [a for a in attractions if '武汉防汛纪念碑' in a['name']]
    other_attractions = [a for a in attractions if '武汉防汛纪念碑' not in a['name']]
    attractions = flood_monument + other_attractions
    
    # 过滤酒店（经济型连锁且评分≥4.4）
    hotels = [h for h in poi_data['accommodations'] if h['type'] == '经济型连锁' and float(h['rating']) >= 4.4]
    
    # 过滤餐厅（必须包含天麻黄芪烧长寿鱼的餐厅）
    restaurants = [r for r in poi_data['restaurants']]
    target_rest = [r for r in restaurants if '天麻黄芪烧长寿鱼' in r['recommended_food']]
    other_rest = [r for r in restaurants if '天麻黄芪烧长寿鱼' not in r['recommended_food']]
    restaurants = target_rest + other_rest

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

    # 约束条件：每天必须选择1个景点
    def attr_selection_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.attr_selection = pyo.Constraint(model.days, rule=attr_selection_rule)

    # 约束条件：景点不重复
    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 约束条件：每天必须选择3个餐厅
    def rest_selection_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.rest_selection = pyo.Constraint(model.days, rule=rest_selection_rule)

    # 约束条件：餐厅不重复（每天不同）
    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_rest = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 约束条件：必须包含武汉防汛纪念碑
    def flood_monument_rule(model):
        flood_ids = [a for a in model.attractions if '武汉防汛纪念碑' in model.attr_data[a]['name']]
        return sum(model.select_attr[d, a] for d in model.days for a in flood_ids) >= 1

    model.flood_monument = pyo.Constraint(rule=flood_monument_rule)

    # 约束条件：必须包含天麻黄芪烧长寿鱼的餐厅
    def target_food_rule(model):
        target_ids = [r for r in model.restaurants if '天麻黄芪烧长寿鱼' in model.rest_data[r]['recommended_food']]
        return sum(model.select_rest[d, r] for d in model.days for r in target_ids) >= 1

    model.target_food = pyo.Constraint(rule=target_food_rule)

    # 约束条件：每日活动时间不超过840分钟
    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants)
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
            for a in model.attractions for h in model.accommodations
        )
        return attr_time + rest_time + transport_time <= 840

    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

    # 约束条件：只选择1个酒店
    def hotel_selection_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.hotel_selection = pyo.Constraint(rule=hotel_selection_rule)

    # 约束条件：只选择1个出发车次和1个返程车次
    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.train_departure_selection = pyo.Constraint(rule=train_departure_rule)
    model.train_back_selection = pyo.Constraint(rule=train_back_rule)

    # 目标函数：最小化成本
    def obj_rule(model):
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
        return total_cost

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

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
        return total_cost <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取出发和返程车次
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = model.train_departure_data[t]
            break
    
    back_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            back_train = model.train_back_data[t]
            break
    
    # 生成每日计划
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'hotel': None,
            'attractions': [],
            'restaurants': [],
            'transport_mode': None,
            'transport_time': 0
        }
        
        if d < travel_days:
            day_plan['hotel'] = selected_hotel['name']
        
        # 获取景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attractions'].append({
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration'],
                    'cost': model.attr_data[a]['cost']
                })
                break
        
        # 获取餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'recommended_food': model.rest_data[r]['recommended_food'],
                    'cost': model.rest_data[r]['cost'],
                    'duration': model.rest_data[r]['duration']
                })
        
        # 获取交通方式
        transport_mode = 'taxi' if pyo.value(model.trans_mode[d]) < 0.5 else 'bus'
        day_plan['transport_mode'] = transport_mode
        
        # 计算交通时间
        if d < travel_days and len(day_plan['attractions']) > 0:
            a_id = day_plan['attractions'][0]['name']
            h_id = selected_hotel['name']
            key1 = f"{h_id},{a_id}"
            key2 = f"{a_id},{h_id}"
            if key1 in intra_city_trans:
                data = intra_city_trans[key1]
                day_plan['transport_time'] += float(data['taxi_duration']) if transport_mode == 'taxi' else float(data['bus_duration'])
            if key2 in intra_city_trans:
                data = intra_city_trans[key2]
                day_plan['transport_time'] += float(data['taxi_duration']) if transport_mode == 'taxi' else float(data['bus_duration'])
        
        plan[f'Day {d}'] = day_plan
    
    # 添加交通信息
    plan['departure_train'] = {
        'train_number': departure_train['train_number'],
        'origin_station': departure_train['origin_station'],
        'destination_station': departure_train['destination_station'],
        'duration': departure_train['duration'],
        'cost': departure_train['cost']
    }
    
    plan['back_train'] = {
        'train_number': back_train['train_number'],
        'origin_station': back_train['origin_station'],
        'destination_station': back_train['destination_station'],
        'duration': back_train['duration'],
        'cost': back_train['cost']
    }
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()