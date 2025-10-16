import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "成都市"
destination_city = "北京市"
budget = 40000
start_date = "2025年04月20日"
end_date = "2025年04月26日"
travel_days = 7
peoples = 2

# 获取数据
def fetch_data():
    url = "http://localhost:12457"
    cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    # 获取景点时过滤评分和价格
    all_attractions = requests.get(url + f"/attractions/{destination_city}").json()
    attractions = [a for a in all_attractions if float(a['rating']) >= 4.0 and float(a['cost']) <= 800]
    
    # 获取酒店时过滤连锁酒店(假设连锁酒店的feature包含"连锁"字样)且价格<=3000
    all_accommodations = requests.get(url + f"/accommodations/{destination_city}").json()
    accommodations = [h for h in all_accommodations if float(h['rating']) >= 4.0 and 
                     float(h['cost']) <= 3000 and "连锁" in h['feature']]
    
    # 获取餐厅时过滤北京菜(假设type包含"北京菜")
    all_restaurants = requests.get(url + f"/restaurants/{destination_city}").json()
    restaurants = [r for r in all_restaurants if "北京菜" in r['type'] and float(r['cost']) <= 300]

    poi_data = {
        'attractions': attractions,
        'accommodations': accommodations,
        'restaurants': restaurants
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
            'destination_id': train_back_dict[t['destination_id']],
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

    # 每日必须选择1个景点
    def daily_attraction_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.daily_attraction = pyo.Constraint(model.days, rule=daily_attraction_rule)

    # 景点不重复
    def unique_attraction_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.unique_attraction = pyo.Constraint(model.attractions, rule=unique_attraction_rule)

    # 每日必须选择3个餐厅
    def daily_restaurant_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.daily_restaurant = pyo.Constraint(model.days, rule=daily_restaurant_rule)

    # 餐厅不重复
    def unique_restaurant_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.unique_restaurant = pyo.Constraint(model.restaurants, rule=unique_restaurant_rule)

    # 只选择1个酒店
    def hotel_selection_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.hotel_selection = pyo.Constraint(rule=hotel_selection_rule)

    # 每日活动时间不超过840分钟
    def time_constraint_rule(model, d):
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
    model.time_constraint = pyo.Constraint(model.days, rule=time_constraint_rule)

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

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        rating = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions) + \
                sum(model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants) + \
                sum(model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations) * (travel_days - 1)
        
        cost = sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions) + \
              sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants) + \
              sum(model.select_hotel[h] * model.hotel_data[h]['cost'] for h in model.accommodations) * (travel_days - 1) + \
              sum(
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
                  for h in model.accommodations) + \
              sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure) + \
              sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        
        return -rating * 100 + cost * 0.01  # 评分权重更大
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = []
    date = datetime.strptime(start_date, "%Y年%m月%d日")
    
    # 获取选择的酒店
    selected_hotel = None
    for h in model.accommodations:
        if pyo.value(model.select_hotel[h]) > 0.5:
            selected_hotel = model.hotel_data[h]
            break
    
    # 获取往返火车信息
    departure_train = None
    for t in model.train_departure:
        if pyo.value(model.select_train_departure[t]) > 0.5:
            departure_train = model.train_departure_data[t]
            break
    
    return_train = None
    for t in model.train_back:
        if pyo.value(model.select_train_back[t]) > 0.5:
            return_train = model.train_back_data[t]
            break
    
    # 添加出发日信息
    plan.append({
        'date': date.strftime("%Y年%m月%d日"),
        'type': '出发',
        'description': f"乘坐{departure_train['train_number']}次列车从{origin_city}前往{destination_city}",
        'details': f"出发站: {departure_train['origin_station']}, 到达站: {departure_train['destination_station']}, 历时: {departure_train['duration']}分钟"
    })
    date += timedelta(days=1)
    
    # 添加每日行程
    for d in model.days:
        if d == travel_days:  # 最后一天
            # 获取景点
            selected_attr = None
            for a in model.attractions:
                if pyo.value(model.select_attr[d, a]) > 0.5:
                    selected_attr = model.attr_data[a]
                    break
            
            # 获取餐厅
            selected_rests = []
            for r in model.restaurants:
                if pyo.value(model.select_rest[d, r]) > 0.5:
                    selected_rests.append(model.rest_data[r])
            
            # 获取交通方式
            transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "出租车"
            
            plan.append({
                'date': date.strftime("%Y年%m月%d日"),
                'type': '游玩',
                'description': f"游览景点: {selected_attr['name']}",
                'details': f"评分: {selected_attr['rating']}, 游玩时间: {selected_attr['duration']}分钟, 费用: {selected_attr['cost']}元",
                'restaurants': [f"{r['name']}(评分: {r['rating']}, 人均: {r['cost']}元)" for r in selected_rests],
                'transport': f"从酒店到景点使用{transport_mode}"
            })
            
            # 添加返程信息
            plan.append({
                'date': date.strftime("%Y年%m月%d日"),
                'type': '返程',
                'description': f"乘坐{return_train['train_number']}次列车从{destination_city}返回{origin_city}",
                'details': f"出发站: {return_train['origin_station']}, 到达站: {return_train['destination_station']}, 历时: {return_train['duration']}分钟"
            })
        else:
            # 获取景点
            selected_attr = None
            for a in model.attractions:
                if pyo.value(model.select_attr[d, a]) > 0.5:
                    selected_attr = model.attr_data[a]
                    break
            
            # 获取餐厅
            selected_rests = []
            for r in model.restaurants:
                if pyo.value(model.select_rest[d, r]) > 0.5:
                    selected_rests.append(model.rest_data[r])
            
            # 获取交通方式
            transport_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "出租车"
            
            plan.append({
                'date': date.strftime("%Y年%m月%d日"),
                'type': '游玩',
                'description': f"游览景点: {selected_attr['name']}",
                'details': f"评分: {selected_attr['rating']}, 游玩时间: {selected_attr['duration']}分钟, 费用: {selected_attr['cost']}元",
                'restaurants': [f"{r['name']}(评分: {r['rating']}, 人均: {r['cost']}元)" for r in selected_rests],
                'transport': f"从酒店到景点使用{transport_mode}",
                'hotel': f"{selected_hotel['name']}(评分: {selected_hotel['rating']}, 价格: {selected_hotel['cost']}元/晚)"
            })
            date += timedelta(days=1)
    
    return json.dumps(plan, ensure_ascii=False, indent=2)

# 主程序
if __name__ == "__main__":
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("未找到最优解")