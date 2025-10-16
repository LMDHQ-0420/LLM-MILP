import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "重庆市"
destination_city = "南京市"
budget = 30000
start_date = "2025年06月15日"
end_date = "2025年06月17日"
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

    days = list(range(1, travel_days + 1))
    model.days = pyo.Set(initialize=days)

    # 过滤景点：只保留中山陵、中山门、栖霞山等热门景点
    filtered_attractions = [a for a in poi_data['attractions'] 
                          if a['name'] in ['中山陵', '中山门', '栖霞山'] or float(a['rating']) >= 4.5]
    attraction_dict = {a['id']: a for a in filtered_attractions}
    
    # 过滤酒店：评分4.5以上，价格低于600元
    filtered_hotels = [h for h in poi_data['accommodations'] 
                      if float(h['rating']) >= 4.5 and float(h['cost']) <= 600]
    hotel_dict = {h['id']: h for h in filtered_hotels}
    
    # 过滤餐厅：推荐川菜和小吃
    filtered_restaurants = [r for r in poi_data['restaurants'] 
                           if '川菜' in r['type'] or '小吃' in r['type'] or '盐水鸭' in r['recommended_food']]
    restaurant_dict = {r['id']: r for r in filtered_restaurants}
    
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}

    model.attractions = pyo.Set(initialize=attraction_dict.keys())
    model.accommodations = pyo.Set(initialize=hotel_dict.keys())
    model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
    model.train_departure = pyo.Set(initialize=train_departure_dict.keys())
    model.train_back = pyo.Set(initialize=train_back_dict.keys())

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
            'destination_station': train_back_dict[t['destination_station']
        }
    )

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

    # 目标函数：最大化评分
    def obj_rule(model):
        attraction_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating']
                              for d in model.days for a in model.attractions)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating']
                          for h in model.accommodations)
        restaurant_score = sum(model.select_rest[d, r] * model.rest_data[r]['rating']
                              for d in model.days for r in model.restaurants)
        return attraction_score + hotel_score + restaurant_score

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # 约束条件
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    def hotel_selection_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    def train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    def time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] 
                       for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time'])
                       for r in model.restaurants)
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
        return attr_time + rest_time + transport_time <= 840

    def restaurant_cost_rule(model):
        return sum(
            model.select_rest[d, r] * model.rest_data[r]['cost']
            for d in model.days for r in model.restaurants
        ) <= 200 * 3 * peoples * travel_days

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

    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)
    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)
    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)
    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)
    model.hotel_selection = pyo.Constraint(rule=hotel_selection_rule)
    model.train_departure = pyo.Constraint(rule=train_departure_rule)
    model.train_back = pyo.Constraint(rule=train_back_rule)
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)
    model.restaurant_cost = pyo.Constraint(rule=restaurant_cost_rule)
    model.budget_constraint = pyo.Constraint(rule=budget_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = []
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取出发和返程火车
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
            'day': d,
            'attraction': None,
            'restaurants': [],
            'transport_mode': None,
            'hotel': selected_hotel if d != travel_days else None
        }
        
        # 获取当天景点
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                day_plan['attraction'] = model.attr_data[a]
                break
        
        # 获取当天餐厅
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                day_plan['restaurants'].append(model.rest_data[r])
        
        # 获取交通方式
        day_plan['transport_mode'] = '公交' if pyo.value(model.trans_mode[d]) > 0.5 else '出租车'
        
        plan.append(day_plan)
    
    # 计算总费用
    total_cost = 0
    
    # 酒店费用
    hotel_cost = selected_hotel['cost'] * (travel_days - 1) * ((peoples + 1) // 2)
    
    # 景点费用
    attraction_cost = 0
    for d in model.days:
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attraction_cost += model.attr_data[a]['cost']
    attraction_cost *= peoples
    
    # 餐厅费用
    restaurant_cost = 0
    for d in model.days:
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                restaurant_cost += model.rest_data[r]['cost']
    restaurant_cost *= peoples
    
    # 市内交通费用
    transport_cost = 0
    for d in model.days:
        for a in model.attractions:
            for h in model.accommodations:
                if pyo.value(model.attr_hotel[d, a, h]) > 0.5:
                    if pyo.value(model.trans_mode[d]) > 0.5:  # 公交
                        transport_cost += peoples * (
                            get_trans_params(intra_city_trans, h, a, 'bus_cost') + 
                            get_trans_params(intra_city_trans, a, h, 'bus_cost')
                        )
                    else:  # 出租车
                        transport_cost += (
                            get_trans_params(intra_city_trans, h, a, 'taxi_cost') + 
                            get_trans_params(intra_city_trans, a, h, 'taxi_cost')
                        )
    
    # 火车费用
    train_cost = peoples * (departure_train['cost'] + back_train['cost'])
    
    total_cost = hotel_cost + attraction_cost + restaurant_cost + transport_cost + train_cost
    
    # 计算总评分
    total_rating = 0
    
    # 酒店评分
    hotel_rating = selected_hotel['rating']
    
    # 景点评分
    attraction_rating = 0
    for d in model.days:
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                attraction_rating += model.attr_data[a]['rating']
    
    # 餐厅评分
    restaurant_rating = 0
    for d in model.days:
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                restaurant_rating += model.rest_data[r]['rating']
    
    total_rating = hotel_rating + attraction_rating + restaurant_rating
    
    result = {
        'start_date': start_date,
        'end_date': end_date,
        'origin_city': origin_city,
        'destination_city': destination_city,
        'peoples': peoples,
        'departure_train': departure_train,
        'back_train': back_train,
        'daily_plans': plan,
        'total_cost': total_cost,
        'total_rating': total_rating,
        'budget': budget
    }
    
    return json.dumps(result, indent=2, ensure_ascii=False)

# 主程序
def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    main()