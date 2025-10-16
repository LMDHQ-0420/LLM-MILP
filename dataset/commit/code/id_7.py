import json
import pyomo.environ as pyo
import requests

# 用户输入
origin_city = "南京市"
destination_city = "杭州市"
budget = 0  # 不限制预算
start_date = "2025年06月01日"
end_date = "2025年06月03日"
travel_days = 3
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

    # 过滤景点（西湖、灵隐寺、宋城等热门景点）
    attractions = [a for a in poi_data['attractions'] if a['name'] in ['西湖', '灵隐寺', '宋城']]
    # 过滤酒店（评分4.6以上的经济型连锁酒店）
    accommodations = [h for h in poi_data['accommodations'] if h['rating'] >= 4.6 and '经济型' in h['type']]
    # 过滤餐厅（杭帮菜，如楼外楼或知味观）
    restaurants = [r for r in poi_data['restaurants'] if '杭帮菜' in r['type'] or r['name'] in ['楼外楼', '知味观']]

    attraction_dict = {a['id']: a for a in attractions}
    hotel_dict = {h['id']: h for h in accommodations}
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
    model.trans_mode = pyo.Var(model.days, domain=pyo.Binary)  # 0: taxi, 1: bus
    model.select_train_departure = pyo.Var(model.train_departure, domain=pyo.Binary)
    model.select_train_back = pyo.Var(model.train_back, domain=pyo.Binary)
    model.attr_hotel = pyo.Var(model.days, model.attractions, model.accommodations, domain=pyo.Binary)

    # 约束条件：景点与酒店的选择关系
    def link_attr_hotel_rule1(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_attr[d, a]
    def link_attr_hotel_rule2(model, d, a, h):
        return model.attr_hotel[d, a, h] <= model.select_hotel[h]
    def link_attr_hotel_rule3(model, d, a, h):
        return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

    model.link_attr_hotel1 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule1)
    model.link_attr_hotel2 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule2)
    model.link_attr_hotel3 = pyo.Constraint(model.days, model.attractions, model.accommodations, rule=link_attr_hotel_rule3)

    # 每日选择一个景点
    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    # 景点不重复
    def unique_attractions_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions_rule)

    # 每日选择三个餐厅
    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    # 餐厅不重复
    def unique_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants_rule)

    # 选择一家酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 每日活动时间不超过840分钟
    def daily_time_constraint_rule(model, d):
        attr_time = sum(model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions)
        rest_time = sum(model.select_rest[d, r] * model.rest_data[r]['duration'] for r in model.restaurants)
        
        trans_time = 0
        for a in model.attractions:
            for h in model.accommodations:
                trans_time += model.attr_hotel[d, a, h] * (
                    (1 - model.trans_mode[d]) * (get_trans_params(intra_city_trans, h, a, 'taxi_duration') + 
                                               get_trans_params(intra_city_trans, a, h, 'taxi_duration')) +
                    model.trans_mode[d] * (get_trans_params(intra_city_trans, h, a, 'bus_duration') + 
                                         get_trans_params(intra_city_trans, a, h, 'bus_duration'))
                )
        return attr_time + rest_time + trans_time <= 840
    model.daily_time_constraint = pyo.Constraint(model.days, rule=daily_time_constraint_rule)

    # 餐厅人均消费不超过200元
    def restaurant_cost_rule(model):
        return sum(
            model.select_rest[d, r] * model.rest_data[r]['cost']
            for d in model.days for r in model.restaurants
        ) <= 200 * 3 * travel_days
    model.restaurant_cost_constraint = pyo.Constraint(rule=restaurant_cost_rule)

    # 选择一趟去程火车
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    # 选择一趟返程火车
    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 目标函数：最大化评分，最小化成本
    def obj_rule(model):
        total_score = sum(
            model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants
        ) + sum(
            model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations
        )
        
        total_cost = sum(
            model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations
        ) + sum(
            model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants
        ) + sum(
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
        ) + sum(
            model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure
        ) + sum(
            model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back
        )
        
        return -total_score + 0.001 * total_cost  # 主要优化评分，次要优化成本

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = []
    
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
    
    # 添加交通信息
    plan.append(f"出发交通: {selected_train_departure['train_number']} 从 {selected_train_departure['origin_station']} 到 {selected_train_departure['destination_station']}")
    plan.append(f"返程交通: {selected_train_back['train_number']} 从 {selected_train_back['origin_station']} 到 {selected_train_back['destination_station']}")
    plan.append(f"住宿: {selected_hotel['name']} (评分: {selected_hotel['rating']}, 类型: {selected_hotel['type']})")
    plan.append("")
    
    # 每日计划
    for d in model.days:
        day_plan = []
        day_plan.append(f"第{d}天:")
        
        # 获取当天景点
        selected_attr = None
        for a in model.attractions:
            if pyo.value(model.select_attr[d, a]) > 0.5:
                selected_attr = model.attr_data[a]
                break
        
        # 获取当天餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append(model.rest_data[r])
        
        # 获取交通方式
        trans_mode = "公交" if pyo.value(model.trans_mode[d]) > 0.5 else "打车"
        
        # 计算交通时间
        trans_time = 0
        if selected_attr and selected_hotel:
            key = f"{selected_hotel['id']},{selected_attr['id']}"
            if key in intra_city_trans:
                data = intra_city_trans[key]
                trans_time = data['bus_duration'] if trans_mode == "公交" else data['taxi_duration']
        
        day_plan.append(f"景点: {selected_attr['name']} (评分: {selected_attr['rating']}, 游玩时间: {selected_attr['duration']}分钟)")
        day_plan.append(f"交通方式: {trans_mode} (往返时间: {2*trans_time:.1f}分钟)")
        day_plan.append("餐厅:")
        for r in selected_rests:
            day_plan.append(f"  - {r['name']} (推荐菜: {r['recommended_food']}, 评分: {r['rating']})")
        
        plan.append("\n".join(day_plan))
    
    return "\n".join(plan)

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
        print("求解失败")