import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "武汉市"
destination_city = "青岛市"
budget = 12000
start_date = "2025年09月05日"
end_date = "2025年09月08日"
travel_days = 4
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

    # 过滤景点
    required_attractions = ['栈桥', '崂山']
    attractions = [a for a in poi_data['attractions'] if a['name'] in required_attractions]
    hotels = [h for h in poi_data['accommodations'] if h['rating'] >= 4.5]
    restaurants = [r for r in poi_data['restaurants'] if r['cost'] <= 400 and '锅贴' in r.get('recommended_food', '')]

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
            'rating': float(hotel_dict[h]['rating'])
        }
    )

    model.rest_data = pyo.Param(
        model.restaurants,
        initialize=lambda m, r: {
            'id': restaurant_dict[r]['id'],
            'name': restaurant_dict[r]['name'],
            'cost': float(restaurant_dict[r]['cost']),
            'rating': float(restaurant_dict[r]['rating']),
            'duration': float(restaurant_dict[r]['duration'])
        }
    )

    model.train_departure_data = pyo.Param(
        model.train_departure,
        initialize=lambda m, t: {
            'train_number': train_departure_dict[t]['train_number'],
            'cost': float(train_departure_dict[t]['cost']),
            'duration': float(train_departure_dict[t]['duration'])
        }
    )

    model.train_back_data = pyo.Param(
        model.train_back,
        initialize=lambda m, t: {
            'train_number': train_back_dict[t]['train_number'],
            'cost': float(train_back_dict[t]['cost']),
            'duration': float(train_back_dict[t]['duration'])
        }
    )

    # 定义变量
    model.select_attr = pyo.Var(model.days, model.attractions, domain=pyo.Binary)
    model.select_hotel = pyo.Var(model.accommodations, domain=pyo.Binary)
    model.select_rest = pyo.Var(model.days, model.restaurants, domain=pyo.Binary)
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 1 for bus
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)

    # 约束条件
    # 每天一个景点且不重复
    def one_attraction_per_day(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attraction = pyo.Constraint(model.days, rule=one_attraction_per_day)

    # 景点不重复
    def unique_attractions(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions)

    # 每天3个餐厅
    def three_restaurants_per_day(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_restaurants = pyo.Constraint(model.days, rule=three_restaurants_per_day)

    # 餐厅不重复
    def unique_restaurants(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants)

    # 选择一个酒店
    def one_hotel(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel)

    # 选择一趟去程火车
    def one_train_departure(model):
        return sum(model.select_train_departure[t] for t in model.train_departure]) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure)

    # 选择一趟返程火车
    def one_train_back(model):
        return sum(model.select_train_back[t for t in model.train_back]) == 1

    model.one_train_back = pyo.Constraint(rule=one_train_back)

    # 每日活动时间不超过840分钟
    def daily_time_constraint(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants)
        transport_time = 0
        for a in model.attractions:
            for h in model.accommodations:
                transport_time += model.select_attr[d, a] * model.select_hotel[h] * (
                    get_trans_params(intra_city_trans, h, a, 'bus_duration') * 2
                )
        return attr_time + rest_time + transport_time <= 840

    model.daily_time = pyo.Constraint(model.days, rule=daily_time_constraint)

    # 预算约束
    def budget_rule(model):
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                        for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                            for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                            for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.select_attr[d, a] * model.select_hotel[h] * (
                model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_cost') * 2
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                            for t in model.train_back)
        total_cost = hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost
        return total_cost <= budget

    model.budget = pyo.Constraint(rule=budget_rule)

    # 目标函数：最大化评分，尽可能花完预算
    def obj_rule(model):
        hotel_rating = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations)
        attraction_rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating']
                              for d in model.days for a in model.attractions)
        restaurant_rating = sum(model.select_rest[d, r] * model.rest_data[r]['rating']
                              for d in model.days for r in model.restaurants)
        
        hotel_cost = sum(model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1)
                        for h in model.accommodations)
        attraction_cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost']
                            for d in model.days for a in model.attractions)
        restaurant_cost = sum(model.select_rest[d, r] * model.rest_data[r]['cost']
                            for d in model.days for r in model.restaurants)
        transport_cost = sum(
            model.select_attr[d, a] * model.select_hotel[h] * (
                model.trans_mode[d] * get_trans_params(intra_city_trans, h, a, 'bus_cost') * 2
            )
            for d in model.days
            for a in model.attractions
            for h in model.accommodations
        )
        train_departure_cost = sum(model.select_train_departure[t] * model.train_departure_data[t]['cost']
                                for t in model.train_departure)
        train_back_cost = sum(model.select_train_back[t] * model.train_back_data[t]['cost']
                            for t in model.train_back)
        total_cost = hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_departure_cost + train_back_cost
        
        return hotel_rating + attraction_rating + restaurant_rating + 0.01 * total_cost  # 权重倾向于评分

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            'attraction': None,
            'restaurants': [],
            'hotel': None,
            'transport_mode': '公交' if pyo.value(model.trans_mode[d]) == 1 else '打车'
        }
        
        # 景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) == 1:
                day_plan['attraction'] = {
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration'],
                    'cost': model.attr_data[a]['cost']
                }
                break
        
        # 餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) == 1:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'duration': model.rest_data[r]['duration'],
                    'cost': model.rest_data[r]['cost']
                })
        
        # 酒店 (最后一天没有)
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) == 1:
                    day_plan['hotel'] = {
                        'name': model.hotel_data[h]['name'],
                        'cost': model.hotel_data[h]['cost']
                    }
                    break
        
        plan[f'Day {d}'] = day_plan
    
    # 添加火车信息
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) == 1:
            plan['Departure Train'] = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration']
            }
            break
    
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) == 1:
            plan['Return Train'] = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration']
            }
            break
    
    return plan

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{json.dumps(plan, indent=4, ensure_ascii=False)}\n```")
    else:
        print("No optimal solution found")

if __name__ == "__main__":
    main()