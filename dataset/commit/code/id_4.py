import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "北京市"
destination_city = "成都市"
budget = 0  # 不限制预算
start_date = "2025年05月01日"
end_date = "2025年05月05日"
travel_days = 5
peoples = 3  # 我和父母

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

    # 筛选高品质住宿(评分4.8以上)
    high_quality_hotels = [h for h in poi_data['accommodations'] if float(h['rating']) >= 4.8]
    hotel_dict = {h['id']: h for h in high_quality_hotels}
    
    # 筛选指定景点
    target_attractions = ['青羊宫', '成都动物园', '大慈寺']
    filtered_attractions = [a for a in poi_data['attractions'] if a['name'] in target_attractions]
    attraction_dict = {a['id']: a for a in filtered_attractions}
    
    # 筛选地道美食(火锅、串串香、钟水饺)
    target_foods = ['火锅', '串串香', '钟水饺']
    filtered_restaurants = [r for r in poi_data['restaurants'] if any(food in r['type'] for food in target_foods)]
    restaurant_dict = {r['id']: r for r in filtered_restaurants}
    
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

    # 约束条件：每天选择一个景点
    def one_attr_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1

    model.one_attr_per_day = pyo.Constraint(model.days, rule=one_attr_per_day_rule)

    # 约束条件：景点不重复
    def unique_attr_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1

    model.unique_attr = pyo.Constraint(model.attractions, rule=unique_attr_rule)

    # 约束条件：每天选择三个餐厅
    def three_rest_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3

    model.three_rest_per_day = pyo.Constraint(model.days, rule=three_rest_per_day_rule)

    # 约束条件：餐厅不重复
    def unique_rest_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1

    model.unique_rest = pyo.Constraint(model.restaurants, rule=unique_rest_rule)

    # 约束条件：选择一个酒店
    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1

    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    # 约束条件：每天活动时间不超过840分钟
    def time_limit_rule(model, d):
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
            for a in model.attractions
            for h in model.accommodations
        )
        
        return attr_time + rest_time + transport_time <= 840

    model.time_limit = pyo.Constraint(model.days, rule=time_limit_rule)

    # 约束条件：选择去程和返程火车各一班
    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1

    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    # 目标函数：最大化体验(评分)
    def obj_rule(model):
        # 酒店评分(乘以天数-1)
        hotel_score = sum(model.select_hotel[h] * model.hotel_data[h]['rating'] * (travel_days - 1) 
                         for h in model.accommodations)
        
        # 景点评分
        attr_score = sum(model.select_attr[d, a] * model.attr_data[a]['rating'] 
                        for d in model.days for a in model.attractions)
        
        # 餐厅评分(优先选择评分高且排队时间短的)
        rest_score = sum(model.select_rest[d, r] * (model.rest_data[r]['rating'] * 2 - model.rest_data[r]['queue_time'] / 10)
                        for d in model.days for r in model.restaurants)
        
        return -(hotel_score + attr_score + rest_score)  # 最小化负分=最大化正分

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

# 生成每日计划
def generate_daily_plan(model, intra_city_trans):
    plan = {}
    
    # 获取选择的酒店
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
    
    # 获取去程火车信息
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
    
    # 获取返程火车信息
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
    
    # 添加交通信息
    plan['transport'] = {
        'departure': departure_train,
        'back': back_train
    }
    
    # 添加住宿信息
    plan['accommodation'] = selected_hotel
    
    # 添加每日行程
    daily_plans = []
    for d in model.days:
        day_plan = {'day': d}
        
        # 获取当天景点
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
        
        # 获取当天餐厅
        selected_rests = []
        for r in model.restaurants:
            if pyo.value(model.select_rest[d, r]) > 0.5:
                selected_rests.append({
                    'id': model.rest_data[r]['id'],
                    'name': model.rest_data[r]['name'],
                    'cost': model.rest_data[r]['cost'],
                    'type': model.rest_data[r]['type'],
                    'rating': model.rest_data[r]['rating'],
                    'queue_time': model.rest_data[r]['queue_time'],
                    'duration': model.rest_data[r]['duration']
                })
        
        # 获取交通方式
        transport_mode = 'taxi' if pyo.value(model.trans_mode[d]) < 0.5 else 'bus'
        
        # 计算交通时间和费用
        if selected_attr and selected_hotel:
            key1 = f"{selected_hotel['id']},{selected_attr['id']}"
            key2 = f"{selected_attr['id']},{selected_hotel['id']}"
            
            if transport_mode == 'taxi':
                transport_time = intra_city_trans[key1]['taxi_duration'] + intra_city_trans[key2]['taxi_duration']
                transport_cost = (intra_city_trans[key1]['taxi_cost'] + intra_city_trans[key2]['taxi_cost'])
            else:
                transport_time = intra_city_trans[key1]['bus_duration'] + intra_city_trans[key2]['bus_duration']
                transport_cost = peoples * (intra_city_trans[key1]['bus_cost'] + intra_city_trans[key2]['bus_cost'])
        else:
            transport_time = 0
            transport_cost = 0
        
        day_plan['attraction'] = selected_attr
        day_plan['restaurants'] = selected_rests
        day_plan['transport_mode'] = transport_mode
        day_plan['transport_time'] = transport_time
        day_plan['transport_cost'] = transport_cost
        
        daily_plans.append(day_plan)
    
    plan['daily_plans'] = daily_plans
    
    return plan

# 主程序
def main():
    # 获取数据
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    
    # 构建模型
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    # 求解
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    # 生成计划
    plan = generate_daily_plan(model, intra_city_trans)
    
    # 输出计划
    print(f"```generated_plan\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```")

if __name__ == "__main__":
    main()