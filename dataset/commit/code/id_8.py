import json
import pyomo.environ as pyo
from datetime import datetime, timedelta
import requests

# 用户输入
origin_city = "杭州市"
destination_city = "南京市"
budget = 0  # 不限制预算
start_date = "2025年04月18日"
end_date = "2025年04月22日"
travel_days = 5
peoples = 4

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

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations'] if h['cost'] <= 1200}
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}
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
            'destination_station': train_back_dict[t]['destination_station']
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

    def obj_rule(model):
        total_rating = sum(
            model.select_attr[d, a] * model.attr_data[a]['rating'] for d in model.days for a in model.attractions
        ) + sum(
            model.select_rest[d, r] * model.rest_data[r]['rating'] for d in model.days for r in model.restaurants
        ) + sum(
            model.select_hotel[h] * model.hotel_data[h]['rating'] for h in model.accommodations
        )
        
        total_cost = sum(
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
            for h in model.accommodations
        ) + (peoples + 1) // 2 * sum(
            model.select_hotel[h] * model.hotel_data[h]['cost'] * (travel_days - 1) for h in model.accommodations
        ) + peoples * (
            sum(model.select_attr[d, a] * model.attr_data[a]['cost'] for d in model.days for a in model.attractions) +
            sum(model.select_rest[d, r] * model.rest_data[r]['cost'] for d in model.days for r in model.restaurants) +
            sum(model.select_train_departure[t] * model.train_departure_data[t]['cost'] for t in model.train_departure) +
            sum(model.select_train_back[t] * model.train_back_data[t]['cost'] for t in model.train_back)
        )
        
        return -total_rating + 0.001 * total_cost  # 最大化评分，同时最小化成本

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def one_attraction_per_day_rule(model, d):
        return sum(model.select_attr[d, a] for a in model.attractions) == 1
    model.one_attraction_per_day = pyo.Constraint(model.days, rule=one_attraction_per_day_rule)

    def unique_attractions_rule(model, a):
        return sum(model.select_attr[d, a] for d in model.days) <= 1
    model.unique_attractions = pyo.Constraint(model.attractions, rule=unique_attractions_rule)

    def three_restaurants_per_day_rule(model, d):
        return sum(model.select_rest[d, r] for r in model.restaurants) == 3
    model.three_restaurants_per_day = pyo.Constraint(model.days, rule=three_restaurants_per_day_rule)

    def unique_restaurants_rule(model, r):
        return sum(model.select_rest[d, r] for d in model.days) <= 1
    model.unique_restaurants = pyo.Constraint(model.restaurants, rule=unique_restaurants_rule)

    def one_hotel_rule(model):
        return sum(model.select_hotel[h] for h in model.accommodations) == 1
    model.one_hotel = pyo.Constraint(rule=one_hotel_rule)

    def one_train_departure_rule(model):
        return sum(model.select_train_departure[t] for t in model.train_departure) == 1
    model.one_train_departure = pyo.Constraint(rule=one_train_departure_rule)

    def one_train_back_rule(model):
        return sum(model.select_train_back[t] for t in model.train_back) == 1
    model.one_train_back = pyo.Constraint(rule=one_train_back_rule)

    def daily_time_constraint_rule(model, d):
        if d == 1:
            # 第一天只有下午活动
            time_limit = 420
        elif d == travel_days:
            # 最后一天只有上午活动
            time_limit = 420
        else:
            time_limit = 840
            
        attraction_time = sum(
            model.select_attr[d, a] * model.attr_data[a]['duration'] for a in model.attractions
        )
        restaurant_time = sum(
            model.select_rest[d, r] * (model.rest_data[r]['duration'] + model.rest_data[r]['queue_time']) 
            for r in model.restaurants
        )
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
        return attraction_time + restaurant_time + transport_time <= time_limit
    model.daily_time_constraint = pyo.Constraint(model.days, rule=daily_time_constraint_rule)

    return model

def generate_daily_plan(model, intra_city_trans):
    plan = []
    
    # 获取出发和返回火车信息
    departure_train = next(t for t in model.train_departure if model.select_train_departure[t].value == 1)
    back_train = next(t for t in model.train_back if model.select_train_back[t].value == 1)
    
    # 获取酒店信息
    hotel_id = next(h for h in model.accommodations if model.select_hotel[h].value == 1)
    hotel = model.hotel_data[hotel_id]
    
    plan.append(f"出发日期: {start_date}")
    plan.append(f"返回日期: {end_date}")
    plan.append(f"出发火车: {model.train_departure_data[departure_train]['train_number']} "
                f"从{model.train_departure_data[departure_train]['origin_station']}到"
                f"{model.train_departure_data[departure_train]['destination_station']}, "
                f"耗时{model.train_departure_data[departure_train]['duration']}分钟")
    plan.append(f"住宿酒店: {hotel['name']} (评分: {hotel['rating']}, 价格: {hotel['cost']}元/晚)")
    
    for d in model.days:
        day_plan = []
        day_plan.append(f"\n第{d}天:")
        
        # 获取当天景点
        attr_id = next(a for a in model.attractions if model.select_attr[d, a].value == 1)
        attr = model.attr_data[attr_id]
        
        # 获取当天餐厅
        rest_ids = [r for r in model.restaurants if model.select_rest[d, r].value == 1]
        rests = [model.rest_data[r] for r in rest_ids]
        
        # 获取交通方式
        transport_mode = "出租车" if model.trans_mode[d].value == 0 else "公交车"
        
        day_plan.append(f"景点: {attr['name']} (评分: {attr['rating']}, 游玩时间: {attr['duration']}分钟)")
        day_plan.append("餐厅:")
        for r in rests:
            day_plan.append(f"  - {r['name']} (评分: {r['rating']}, 推荐菜: {r['recommended_food']})")
        day_plan.append(f"交通方式: {transport_mode}")
        
        plan.extend(day_plan)
    
    plan.append(f"\n返程火车: {model.train_back_data[back_train]['train_number']} "
                f"从{model.train_back_data[back_train]['origin_station']}到"
                f"{model.train_back_data[back_train]['destination_station']}, "
                f"耗时{model.train_back_data[back_train]['duration']}分钟")
    
    return "\n".join(plan)

def main():
    cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data()
    model = build_model(cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans)
    
    solver = pyo.SolverFactory('scip')
    results = solver.solve(model, tee=True)
    
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        plan = generate_daily_plan(model, intra_city_trans)
        print(f"```generated_plan\n{plan}\n```")
    else:
        print("优化失败，无法生成旅行计划")

if __name__ == "__main__":
    main()