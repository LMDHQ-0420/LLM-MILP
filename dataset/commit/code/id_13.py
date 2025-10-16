import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "洛阳市"
destination_city = "贵阳市"
budget = 5000
start_date = "2025年05月25日"
end_date = "2025年05月27日"
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

    # 筛选4.6分以上且价格低于600的经济型连锁酒店
    poi_data['accommodations'] = [h for h in poi_data['accommodations'] 
                                 if h['rating'] >= 4.6 and h['cost'] < 600 and '经济型' in h['type']]
    
    # 筛选指定景点
    target_attractions = ['黔灵山公园', '马鞍山公园', '天河潭']
    poi_data['attractions'] = [a for a in poi_data['attractions'] if a['name'] in target_attractions]
    
    # 筛选贵州特色餐厅
    target_foods = ['肠旺面', '酸汤鱼']
    poi_data['restaurants'] = [r for r in poi_data['restaurants'] 
                              if any(food in r['recommended_food'] for food in target_foods)]

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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 0: taxi, 1: bus
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

    # 约束条件：每天一个景点且不重复
    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    def unique_attractions_rule(model):
        return sum(model.select_attr[d, a] for d in model.days for a in model.attractions) == len(model.attractions)

    model.unique_attractions = pyo.Constraint(rule=unique_attractions_rule)

    # 约束条件：每天3个餐厅
    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    # 约束条件：餐厅不重复
    def unique_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants_rule)

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

    # 约束条件：选择一次去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    # 约束条件：选择一次返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 约束条件：公共交通为主
    def public_transport_rule(model, d):
        return model.trans_mode[d] == 1  # 强制使用公交

    model.public_transport = pyo.Constraint(model.days, rule=public_transport_rule)

    # 约束条件：人均餐饮消费不超过300元
    def restaurant_budget_rule(model):
        return sum(
            model.select_rest[d, r] * model.rest_data[r]['cost']
            for d in model.days for r in model.restaurants
        ) <= 300 * 3 * peoples * travel_days

    model.restaurant_budget = pyo.Constraint(rule=restaurant_budget_rule)

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        total_rating = sum(
            model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants
        ) + sum(
            model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations
        )
        
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
        
        return -total_rating + 0.001 * total_cost  # 优先评分，次要考虑成本

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

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    days_plan = {}
    
    # 获取酒店信息
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = {
                'id': model.hotel_data[h]['id'],
                'name': model.hotel_data[h]['name'],
                'cost': model.hotel_data[h]['cost'],
                'rating': model.hotel_data[h]['rating']
            }
            break
    
    # 获取火车信息
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration'],
                'origin_station': model.train_departure_data[t]['origin_station'],
                'destination_station': model.train_departure_data[t]['destination_station']
            }
            break
    
    back_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            back_train = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration'],
                'origin_station': model.train_back_data[t]['origin_station'],
                'destination_station': model.train_back_data[t]['destination_station']
            }
            break
    
    # 生成每日计划
    for d in model.days:
        # 获取景点
        selected_attr = None
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                selected_attr = {
                    'id': model.attr_data[a]['id'],
                    'name': model.attr_data[a]['name'],
                    'cost': model.attr_data[a]['cost'],
                    'duration': model.attr_data[a]['duration'],
                    'rating': model.attr_data[a]['rating']
                }
                break
        
        # 获取餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append({
                    'id': model.rest_data[r]['id'],
                    'name': model.rest_data[r]['name'],
                    'cost': model.rest_data[r]['cost'],
                    'duration': model.rest_data[r]['duration'],
                    'rating': model.rest_data[r]['rating'],
                    'recommended_food': model.rest_data[r]['recommended_food']
                })
        
        # 获取交通方式
        transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "出租车"
        
        # 计算交通时间
        if selected_attr and selected_hotel:
            trans_time = get_trans_params(intra_city_trans, selected_hotel['id'], selected_attr['id'], 'bus_duration')
        else:
            trans_time = 0
        
        days_plan[f"Day {d}"] = {
            'attraction': selected_attr,
            'restaurants': selected_rests,
            'hotel': selected_hotel if d < travel_days else None,
            'transport_mode': transport_mode,
            'transport_time': trans_time * 2  # 往返
        }
    
    # 计算总成本
    total_cost = 0
    
    # 火车费用
    if departure_train:
        total_cost += peoples * departure_train['cost']
    if back_train:
        total_cost += peoples * back_train['cost']
    
    # 酒店费用
    if selected_hotel:
        total_cost += ((peoples + 1) // 2) * selected_hotel['cost'] * (travel_days - 1)
    
    # 景点费用
    for d in days_plan.values():
        if d['attraction']:
            total_cost += peoples * d['attraction']['cost']
    
    # 餐饮费用
    for d in days_plan.values():
        for r in d['restaurants']:
            total_cost += peoples * r['cost']
    
    # 市内交通费用
    for d in days_plan.values():
        if d['transport_mode'] == '公交' and d['attraction'] and selected_hotel:
            cost = get_trans_params(intra_city_trans, selected_hotel['id'], d['attraction']['id'], 'bus_cost')
            total_cost += peoples * cost * 2  # 往返
    
    plan = {
        'origin_city': origin_city,
        'destination_city': destination_city,
        'start_date': start_date,
        'end_date': end_date,
        'peoples': peoples,
        'total_cost': total_cost,
        'departure_train': departure_train,
        'back_train': back_train,
        'daily_plans': days_plan
    }
    
    return json.dumps(plan, indent=2, ensure_ascii=False)

# 主函数
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("Optimization successful!")
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("Optimization failed. No feasible solution found.")

if __name__ == "__main__":
    main()