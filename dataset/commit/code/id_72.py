import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "重庆市"
destination_city = "杭州市"
budget = 13500
start_date = "2025年10月15日"
end_date = "2025年10月19日"
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

    # 筛选景点：必须包含拱宸桥和华家池
    attractions = poi_data['attractions']
    required_attractions = ['拱宸桥', '华家池']
    filtered_attractions = [a for a in attractions if a['name'] in required_attractions or float(a['rating']) >= 4.5]
    attraction_dict = {a['id']: a for a in filtered_attractions}

    # 筛选住宿：评分4.8以上且类型为民宿
    accommodations = [h for h in poi_data['accommodations'] if float(h['rating']) >= 4.8 and '民宿' in h['type']]
    hotel_dict = {h['id']: h for h in accommodations}

    # 筛选餐厅：包含叫化鸡且人均消费<=300
    restaurants = [r for r in poi_data['restaurants'] 
                  if ('叫化鸡' in r.get('recommended_food', '') or float(r['rating']) >= 4.5) 
                  and float(r['cost']) <= 300]
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
            'destination_station': train_back_dict[t['destination_station']]
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

    # 每日必须选择三个餐厅
    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    # 每日活动时间不超过840分钟
    def daily_time_rule(model, d):
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

    model.daily_time = pyo.Constraint(model.days, rule=daily_time_rule)

    # 必须选择拱宸桥和华家池
    def required_attractions_rule(model):
        required_names = ['拱宸桥', '华家池']
        required_attractions = [a for a in model.attractions if model.attr_data[a]['name'] in required_names]
        return sum(model.select_attr[d, a] for d in model.days for a in required_attractions) == len(required_names)

    model.required_attractions = pyo.Constraint(rule=required_attractions_rule)

    # 必须选择一家民宿
    def select_one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.select_one_hotel = pyo.Constraint(rule=select_one_hotel_rule)

    # 必须选择往返火车
    def select_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def select_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.select_train_departure_con = pyo.Constraint(rule=select_train_departure_rule)
    model.select_train_back_con = pyo.Constraint(rule=select_train_back_rule)

    # 目标函数：最小化市内交通时间
    def obj_rule(model):
        return sum(
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
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )

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
        return (peoples+1) // 2 * hotel_cost + transport_cost + peoples * (
                     attraction_cost + restaurant_cost + train_departure_cost + train_back_cost) <= budget

    model.budget_con = pyo.Constraint(rule=budget_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = ""
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break

    train_departure = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            train_departure = model.train_departure_data[t]
            break

    train_back = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            train_back = model.train_back_data[t]
            break

    plan += f"出发火车: {train_departure['train_number']} 从 {train_departure['origin_station']} 到 {train_departure['destination_station']}\n"
    plan += f"返程火车: {train_back['train_number']} 从 {train_back['origin_station']} 到 {train_back['destination_station']}\n"
    plan += f"住宿: {selected_hotel['name']} (评分: {selected_hotel['rating']}, 类型: {selected_hotel['type']})\n\n"

    for d in model.days:
        plan += f"第{d}天:\n"
        
        # 景点
        selected_attr = None
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                selected_attr = model.attr_data[a]
                break
        plan += f"  景点: {selected_attr['name']} (评分: {selected_attr['rating']}, 游玩时间: {selected_attr['duration']}分钟)\n"

        # 餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append(model.rest_data[r])
        plan += "  餐厅:\n"
        for r in selected_rests:
            plan += f"    {r['name']} (推荐: {r['recommended_food']}, 人均: {r['cost']}元)\n"

        # 交通方式
        trans_mode = "打车" if pyo.value(model.trans_mode[d]) < 0.5 else "公交"
        plan += f"  市内交通: {trans_mode}\n"

        # 计算时间
        attr_time = selected_attr['duration']
        rest_time = sum(r['duration'] + r['queue_time'] for r in selected_rests)
        
        trans_time = 0
        for a in model.attractions:
            for h in model.accommodations:
                if pyo.value(model.attr_hotel[d, a, h]) > 0.5:
                    if pyo.value(model.trans_mode[d]) < 0.5:  # 打车
                        trans_time += get_trans_params(intra_city_trans, h['id'], a['id'], 'taxi_duration') + \
                                    get_trans_params(intra_city_trans, a['id'], h['id'], 'taxi_duration')
                    else:  # 公交
                        trans_time += get_trans_params(intra_city_trans, h['id'], a['id'], 'bus_duration') + \
                                    get_trans_params(intra_city_trans, a['id'], h['id'], 'bus_duration')
                    break
        
        total_time = attr_time + rest_time + trans_time
        plan += f"  总活动时间: {total_time}分钟\n\n"

    return plan

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)

    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No solution found")