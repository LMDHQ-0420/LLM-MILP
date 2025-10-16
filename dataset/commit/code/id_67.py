import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "武汉市"
destination_city = "三亚市"
budget = 12000
start_date = "2025年06月22日"
end_date = "2025年06月25日"
travel_days = 4
peoples = 1

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 获取所有POI数据
    all_attractions = requests.get(url + f"/attractions/{destination_city}").json()
    all_accommodations = requests.get(url + f"/accommodations/{destination_city}").json()
    all_restaurants = requests.get(url + f"/restaurants/{destination_city}").json()

    # 根据要求筛选数据
    attractions = [a for a in all_attractions if a['name'] in ['凤凰岭', '热带天堂森林公园']]
    accommodations = [h for h in all_accommodations if float(h['rating']) >= 4.6]
    restaurants = [r for r in all_restaurants if '东山羊' in r['recommended_food'] and float(r['cost']) <= 400]

    poi_data = {
        'attractions': attractions,
        'accommodations': accommodations,
        'restaurants': restaurants
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
            'destination_station': train_departure_dict[t['destination_station']
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

    # 约束条件：每日活动时间不超过840分钟
    def activity_time_rule(model, d):
        if d == travel_days:  # 最后一天不考虑住宿
            return sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions) + \
                   sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants) + \
                   sum(model.attr_hotel[d, a, h] * get_trans_params(intra_city_trans, h, a, 'bus_duration') * 2 
                       for a in model.attractions for h in model.accommodations) <= 840
        else:
            return sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions) + \
                   sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) for r in model.restaurants) + \
                   sum(model.attr_hotel[d, a, h] * get_trans_params(intra_city_trans, h, a, 'bus_duration') * 2 
                       for a in model.attractions for h in model.accommodations) <= 840

    model.activity_time_constraint = pyo.Constraint(model.days, rule=activity_time_rule)

    # 约束条件：每天1个景点且景点不重复
    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.one_attr_per_day = pyo.Constraint(model.days, rule=one_attr_per_day_rule)
    model.unique_attr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 约束条件：每天3个餐厅且餐厅不重复
    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.three_rest_per_day = pyo.Constraint(model.days, rule=three_rest_per_day_rule)
    model.unique_rest = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 约束条件：只选一个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 约束条件：交通方式以公交为主
    def transport_mode_rule(model, d):
        return model.trans_mode[d] == 1  # 1表示公交

    model.transport_mode = pyo.Constraint(model.days, rule=transport_mode_rule)

    # 目标函数：最大化评分，最小化交通时间
    def obj_rule(model):
        # 评分部分
        attraction_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                              for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] 
                          for h in model.accommodations)
        restaurant_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating'] 
                              for d in model.days for r in model.restaurants)
        
        # 交通时间部分
        transport_time = sum(
            model.attr_hotel[d, a, h] * get_trans_params(intra_city_trans, h, a, 'bus_duration') * 2
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )
        
        return - (attraction_score + hotel_score + restaurant_score) + 0.1 * transport_time

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
            model.attr_hotel[d, a, h] * peoples * (
                get_trans_params(intra_city_trans, h, a, 'bus_cost') + \
                get_trans_params(intra_city_trans, a, h, 'bus_cost')
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations)
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                   for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                              for t in model.train_back)
        total_cost = hotel_cost + attraction_cost + restaurant_cost + transport_cost + \
                     peoples * (train_departure_cost + train_back_cost)
        return total_cost <= budget

    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    
    # 获取选中的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取选中的火车
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
    
    # 生成每日计划
    for d in model.days:
        day_plan = {
            "date": (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            "hotel": selected_hotel['name'] if d != travel_days else None,
            "attraction": None,
            "restaurants": [],
            "transport_mode": "公交",
            "transport_time": 0
        }
        
        # 获取当天景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan["attraction"] = model.attr_data[a]['name']
                # 计算交通时间
                if selected_hotel and d != travel_days:
                    key = f"{selected_hotel['id']},{model.attr_data[a]['id']}"
                    if key in intra_city_trans:
                        day_plan["transport_time"] = intra_city_trans[key]['bus_duration'] * 2
                break
        
        # 获取当天餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan["restaurants"].append(model.rest_data[r]['name'])
        
        plan[f"第{d}天"] = day_plan
    
    # 添加交通信息
    plan["交通"] = {
        "出发": {
            "车次": selected_train_departure['train_number'],
            "出发站": selected_train_departure['origin_station'],
            "到达站": selected_train_departure['destination_station'],
            "出发时间": selected_train_departure['departure_time'],
            "到达时间": selected_train_departure['arrival_time']
        },
        "返程": {
            "车次": selected_train_back['train_number'],
            "出发站": selected_train_back['origin_station'],
            "到达站": selected_train_back['destination_station'],
            "出发时间": selected_train_back['departure_time'],
            "到达时间": selected_train_back['arrival_time']
        }
    }
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
def main():
    # 获取数据
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    
    # 构建模型
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    # 求解
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    # 输出结果
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("未找到最优解")

if __name__ == "__main__":
    main()