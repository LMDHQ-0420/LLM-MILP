```python
import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "厦门市"
destination_city = "贵阳市"
budget = 9000
start_date = "2025年05月22日"
end_date = "2025年05月25日"
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

    # 筛选景点：贵御温泉、天鹅湖景区
    required_attractions = ['贵御温泉', '天鹅湖景区']
    attractions = [a for a in poi_data['attractions'] if a['name'] in required_attractions]
    attraction_dict = {a['id']: a for a in attractions}
    
    # 筛选酒店：评分4.5以上
    hotels = [h for h in poi_data['accommodations'] if h['rating'] >= 4.5]
    hotel_dict = {h['id']: h for h in hotels}
    
    # 筛选餐厅：推荐牛肉粉且人均消费400元内
    restaurants = [r for r in poi_data['restaurants'] 
                  if '牛肉粉' in r.get('recommended_food', '') and r['cost'] <= 400]
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary, initialize=1)  # 默认公交
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

    # 目标函数：最大化景点停留时间，最小化用餐时间
    def obj_rule(model):
        attraction_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                            for d in model.days for a in model.attractions)
        restaurant_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] 
                            for d in model.days for r in model.restaurants)
        return restaurant_time - attraction_time  # 最小化用餐时间，最大化景点时间

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

    # 每日活动时间约束
    def daily_time_rule(model, d):
        if d == 1:  # 第一天不计火车时间
            return sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions) + \
                   sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants) + \
                   sum(model.attr_hotel[d, a, h] * (
                       (1 - model.trans_mode[d]) * (
                           get_trans_params(intra_city_trans, h, a, 'taxi_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'taxi_duration')
                       ) + \
                       model.trans_mode[d] * (
                           get_trans_params(intra_city_trans, h, a, 'bus_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'bus_duration')
                       )
                   ) for a in model.attractions for h in model.accommodations) <= 840
        elif d == travel_days:  # 最后一天不计返程火车时间
            return sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions) + \
                   sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants) + \
                   sum(model.attr_hotel[d, a, h] * (
                       (1 - model.trans_mode[d]) * (
                           get_trans_params(intra_city_trans, h, a, 'taxi_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'taxi_duration')
                       ) + \
                       model.trans_mode[d] * (
                           get_trans_params(intra_city_trans, h, a, 'bus_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'bus_duration')
                       )
                   ) for a in model.attractions for h in model.accommodations) <= 840
        else:
            return sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions) + \
                   sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants) + \
                   sum(model.attr_hotel[d, a, h] * (
                       (1 - model.trans_mode[d]) * (
                           get_trans_params(intra_city_trans, h, a, 'taxi_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'taxi_duration')
                       ) + \
                       model.trans_mode[d] * (
                           get_trans_params(intra_city_trans, h, a, 'bus_duration') + \
                           get_trans_params(intra_city_trans, a, h, 'bus_duration')
                       )
                   ) for a in model.attractions for h in model.accommodations) <= 840

    model.daily_time_con = pyo.Constraint(model.days, rule=daily_time_rule)

    # 每日选择一个景点
    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attr_con = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    # 景点不重复
    def attraction_unique_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.attr_unique_con = pyo.Constraint(model.attractions, rule=attraction_unique_rule)

    # 每日三个餐厅
    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_rest_con = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    # 餐厅不重复
    def restaurant_unique_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.rest_unique_con = pyo.Constraint(model.restaurants, rule=restaurant_unique_rule)

    # 选择一个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel_con = pyo.Constraint(rule=one_hotel_rule)

    # 选择一个去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    model.one_train_departure_con = pyo.Constraint(rule=one_train_departure_rule)

    # 选择一个返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_back_con = pyo.Constraint(rule=one_train_back_rule)

    # 交通方式约束：以公交为主
    def bus_priority_rule(model, d):
        return model.trans_mode[d] == 1

    model.bus_priority_con = pyo.Constraint(model.days, rule=bus_priority_rule)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    for d in model.days:
        day_plan = {
            'date': (datetime.strptime(start_date, "%Y年%m月%d日") + timedelta(days=d-1)).strftime("%Y年%m月%d日"),
            'attraction': None,
            'restaurants': [],
            'hotel': None,
            'transport_mode': []
        }
        
        # 景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attraction'] = {
                    'name': model.attr_data[a]['name'],
                    'duration': model.attr_data[a]['duration'],
                    'cost': model.attr_data[a]['cost']
                }
                break
                
        # 餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append({
                    'name': model.rest_data[r]['name'],
                    'duration': model.rest_data[r]['duration'],
                    'cost': model.rest_data[r]['cost']
                })
                
        # 酒店 (最后一天不选)
        if d < travel_days:
            for h in model.accommodations:
                if pyo.value(model.select_hotel[h]) > 0.5:
                    day_plan['hotel'] = {
                        'name': model.hotel_data[h]['name'],
                        'cost': model.hotel_data[h]['cost']
                    }
                    break
        
        # 交通方式
        trans_mode = '公交' if pyo.value(model.trans_mode[d]) > 0.5 else '打车'
        day_plan['transport_mode'] = trans_mode
        
        plan[f'Day {d}'] = day_plan
    
    # 火车信息
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            plan['departure_train'] = {
                'train_number': model.train_departure_data[t]['train_number'],
                'cost': model.train_departure_data[t]['cost'],
                'duration': model.train_departure_data[t]['duration']
            }
            break
            
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            plan['return_train'] = {
                'train_number': model.train_back_data[t]['train_number'],
                'cost': model.train_back_data[t]['cost'],
                'duration': model.train_back_data[t]['duration']
            }
            break
    
    return plan

# 主程序
if __name__ == '__main__':
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```")
    else:
        print("No feasible solution found")
```