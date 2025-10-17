from typing import List, Dict, Any, Optional, Literal
from datetime import timedelta, datetime
import argparse

from fastapi import FastAPI, Query, Form, UploadFile, File, Path, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator, ValidationError
import uvicorn
import pymysql
from pymysql.cursors import DictCursor

from baseModels import CrossCityTransport, Attraction, Accommodation, Restaurant, Duration, DurationParams, TrainInfo, \
    City, ApiResponse, EvaluationResult
from error_models import ErrorResponse

import asyncio
import os
import time
import json
from utils.common_utils import load_json_file, save_json_file
import aiofiles  
import zipfile  
import uuid
import shutil

# 自定义异常类
class DatabaseConnectionError(Exception):
    """数据库连接异常"""
    pass


class DataNotFoundError(Exception):
    """数据未找到异常"""
    pass


class InvalidParameterError(Exception):
    """参数验证异常"""
    pass


app = FastAPI(
    title="旅游行程规划",
    description="基础查询api",
    version="1.0.0"
)

db_config = {}


def parse_parameters():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--user', type=str, required=True)
    parser.add_argument('--password', type=str, required=True)
    parser.add_argument('--port', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--api_server_port', type=str, required=True)
    args = parser.parse_args()
    return args

@app.on_event("startup")
async def startup_event():
    global db_config
    args = parse_parameters()
    db_config = {
        "host": args.host,
        "user": args.user,
        "password": args.password,
        "port": int(args.port),
        "database": args.database,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor
    }

    print("数据库配置已初始化:", db_config)

def get_db_connection(db_config):
    """获取数据库连接，并处理可能的连接异常"""
    print(db_config)
    try:
        return pymysql.connect(**db_config)
    except pymysql.Error as e:
        error_code = e.args[0] if e.args else 0
        if error_code == 2003:
            raise DatabaseConnectionError("数据库服务器无法连接")
        elif error_code == 1045:
            raise DatabaseConnectionError("数据库认证失败")
        elif error_code == 1049:
            raise DatabaseConnectionError("数据库不存在")
        elif error_code == 2006:
            raise DatabaseConnectionError("数据库连接已断开")
        else:
            raise DatabaseConnectionError(f"数据库连接错误: {str(e)}")


# 全局异常处理器
@app.exception_handler(DatabaseConnectionError)
async def database_connection_exception_handler(request: Request, exc: DatabaseConnectionError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Database Connection Error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(DataNotFoundError)
async def data_not_found_exception_handler(request: Request, exc: DataNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Data Not Found",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(InvalidParameterError)
async def invalid_parameter_exception_handler(request: Request, exc: InvalidParameterError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid Parameter",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "请求参数验证失败",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(pymysql.Error)
async def mysql_exception_handler(request: Request, exc: pymysql.Error):
    error_code = exc.args[0] if exc.args else 0
    return JSONResponse(
        status_code=500,
        content={
            "error": "Database Error",
            "message": f"数据库操作失败 (错误代码: {error_code})",
            "details": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "服务器内部错误",
            "details": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


# API接口定义
@app.get("/cross-city-transport/",
         response_model=List[CrossCityTransport],
         responses={
             200: {"description": "成功返回跨城交通方案"},
             400: {"description": "请求参数错误", "model": ErrorResponse},
             404: {"description": "未找到相关交通方案", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse},
             500: {"description": "服务器内部错误", "model": ErrorResponse}
         })
async def get_cross_city_transport(
        origin_city: str = Query(..., description="出发城市", min_length=1, max_length=50),
        destination_city: str = Query(..., description="目的地城市", min_length=1, max_length=50)
):
    """
    获取跨城交通方案

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "城市名称不能为空"}
    - 404: {"error": "Data Not Found", "message": "未找到从北京到上海的交通方案"}
    - 503: {"error": "Database Connection Error", "message": "数据库服务器无法连接"}
    """
    try:
        # 参数验证
        if not origin_city.strip() or not destination_city.strip():
            raise InvalidParameterError("城市名称不能为空")

        if origin_city.strip() == destination_city.strip():
            raise InvalidParameterError("出发城市和目的地城市不能相同")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT origin_id, destination_id, train_number, duration, price as cost, origin_station, destination_station 
            FROM train_info
            WHERE origin_city = %s 
            AND destination_city = %s;
        """, (origin_city.strip(), destination_city))

        result = cursor.fetchall()
        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到从{origin_city}到{destination_city}的交通方案")

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except pymysql.Error as e:
        error_code = e.args[0] if e.args else 0
        raise HTTPException(
            status_code=500,
            detail=f"数据库查询错误 (错误代码: {error_code}): {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/poi-data", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("city_name 不能为空")
@app.get("/poi-data/{city_name}",
         responses={
             200: {"description": "成功返回POI数据"},
             400: {"description": "城市名称参数错误", "model": ErrorResponse},
             404: {"description": "未找到该城市的POI数据", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse},
             500: {"description": "服务器内部错误", "model": ErrorResponse}
         })
async def get_poi_data(city_name: str = Path(..., description="城市名称", min_length=1, max_length=50)):
    """
    获取市内POI数据

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "城市名称格式不正确"}
    - 404: {"error": "Data Not Found", "message": "未找到城市'XXX'的POI数据"}
    """
    try:
        if not city_name.strip():
            raise InvalidParameterError("城市名称不能为空")

        # 简单的城市名称格式验证
        if len(city_name.strip()) < 2:
            raise InvalidParameterError("城市名称格式不正确")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        # 获取景点数据
        cursor.execute("""
            SELECT attraction_id as id, attraction_name as name, avg_consumption as cost,
                   attraction_type as type, rating, suggested_duration as duration
            FROM poi_attraction 
            WHERE city_name = %s
        """, (city_name,))
        attractions = cursor.fetchall()

        # 获取酒店数据
        cursor.execute("""
            SELECT accommodation_id as id, accommodation_name as name, avg_price as cost,
                   accommodation_type as type, rating, feature_hotel_type as feature
            FROM poi_accommodation 
            WHERE city_name = %s
        """, (city_name,))
        accommodations = cursor.fetchall()

        # 获取餐厅数据
        cursor.execute("""
            SELECT restaurant_id as id, restaurant_name as name, avg_price as cost, rating,
                restaurant_type as type, recommended_food, queue_time, consumption_time as duration
            FROM poi_restaurant 
            WHERE city_name = %s 
        """, (city_name,))
        restaurants = cursor.fetchall()

        conn.close()

        # 检查是否有任何数据
        if not attractions and not accommodations and not restaurants:
            raise DataNotFoundError(f"未找到城市'{city_name}'的POI数据")

        return {
            "attractions": attractions,
            "accommodations": accommodations,
            "restaurants": restaurants
        }

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取POI数据失败: {str(e)}")

@app.get("/attractions", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("city_name 不能为空")
@app.get("/attractions/{city_name}",
         response_model=List[Attraction],
         responses={
             200: {"description": "成功返回景点数据"},
             400: {"description": "请求参数错误", "model": ErrorResponse},
             404: {"description": "未找到景点数据", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
async def get_attractions(
        city_name: str = Path(..., description="城市名称", min_length=1, max_length=50),
        limit: int = Query(None, description="返回数据条数", gt=0, le=100)
):
    """
    获取景点数据

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "limit参数必须大于0且小于等于100"}
    - 404: {"error": "Data Not Found", "message": "未找到城市'XXX'的景点数据"}
    """
    try:
        if not city_name.strip():
            raise InvalidParameterError("城市名称不能为空")

        if limit is not None and (limit <= 0 or limit > 100):
            raise InvalidParameterError("limit参数必须大于0且小于等于100")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        # 获取景点数据

        if limit is None:
            cursor.execute("""
                SELECT attraction_id as id, attraction_name as name, avg_consumption as cost,
                           attraction_type as type, rating, suggested_duration as duration
                FROM poi_attraction 
                WHERE city_name = %s 
            """, (city_name.strip(),))
        else:
            cursor.execute("""
                SELECT attraction_id as id, attraction_name as name, avg_consumption as cost,
                           attraction_type as type, rating, suggested_duration as duration
                FROM poi_attraction 
                WHERE city_name = %s LIMIT %s
            """, (city_name.strip(), limit))

        result = cursor.fetchall()
        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到城市'{city_name}'的景点数据")

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取景点数据失败: {str(e)}")

@app.get("/accommodations", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("city_name 不能为空")
@app.get("/accommodations/{city_name}",
         response_model=List[Accommodation],
         responses={
             200: {"description": "成功返回酒店数据"},
             400: {"description": "请求参数错误", "model": ErrorResponse},
             404: {"description": "未找到酒店数据", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
async def get_accommodations(
        city_name: str = Path(..., description="城市名称", min_length=1, max_length=50),
        limit: int = Query(None, description="返回数据条数", gt=0, le=100)
):
    """
    获取酒店数据

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "limit参数超出允许范围"}
    - 404: {"error": "Data Not Found", "message": "未找到城市'XXX'的酒店数据"}
    """
    try:
        if not city_name.strip():
            raise InvalidParameterError("城市名称不能为空")

        if limit is not None and (limit <= 0 or limit > 100):
            raise InvalidParameterError("limit参数必须大于0且小于等于100")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        if limit is None:
            cursor.execute("""
                SELECT accommodation_id as id, accommodation_name as name, avg_price as cost,
                          accommodation_type as type, rating, feature_hotel_type as feature
                FROM poi_accommodation 
                WHERE city_name = %s
            """, (city_name.strip(),))
        else:
            cursor.execute("""
                SELECT accommodation_id as id, accommodation_name as name, avg_price as cost,
                          accommodation_type as type, rating, feature_hotel_type as feature
                FROM poi_accommodation 
                WHERE city_name = %s LIMIT %s
            """, (city_name.strip(), limit))

        result = cursor.fetchall()
        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到城市'{city_name}'的酒店数据")

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取酒店数据失败: {str(e)}")

@app.get("/restaurants", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("city_name 不能为空")
@app.get("/restaurants/{city_name}",
         response_model=List[Restaurant],
         responses={
             200: {"description": "成功返回餐厅数据"},
             400: {"description": "请求参数错误", "model": ErrorResponse},
             404: {"description": "未找到餐厅数据", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
async def get_restaurants(
        city_name: str = Path(..., description="城市名称", min_length=1, max_length=50),
        limit: int = Query(None, description="返回数据条数", gt=0, le=100)
):
    """
    获取餐厅数据

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "城市名称长度不符合要求"}
    - 404: {"error": "Data Not Found", "message": "未找到城市'XXX'的餐厅数据"}
    """
    try:
        if not city_name.strip():
            raise InvalidParameterError("城市名称不能为空")

        if limit is not None and (limit <= 0 or limit > 100):
            raise InvalidParameterError("limit参数必须大于0且小于等于100")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        if limit is None:
            cursor.execute("""
                SELECT restaurant_id as id, restaurant_name as name, avg_price as cost, rating,
                       restaurant_type as type, recommended_food, queue_time, consumption_time as duration
                FROM poi_restaurant 
                WHERE city_name = %s
            """, (city_name.strip(),))
        else:
            cursor.execute("""
                SELECT restaurant_id as id, restaurant_name as name, avg_price as cost, rating,
                       restaurant_type as type, recommended_food, queue_time, consumption_time as duration
                FROM poi_restaurant 
                WHERE city_name = %s LIMIT %s
            """, (city_name.strip(), limit))

        result = cursor.fetchall()
        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到城市'{city_name}'的餐厅数据")

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取餐厅数据失败: {str(e)}")

@app.get("/intra-city-transport", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("city_name 不能为空")
@app.get("/intra-city-transport/{city_name}",
         responses={
             200: {"description": "成功返回市内交通数据"},
             400: {"description": "城市名称参数错误", "model": ErrorResponse},
             404: {"description": "未找到交通数据", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
def get_intra_city_transport(
        city_name: str = Path(..., description="城市名称", min_length=1, max_length=50)
):
    """
    获取市内交通方案数据

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "城市名称不能为空"}
    - 404: {"error": "Data Not Found", "message": "未找到城市'XXX'的市内交通数据"}
    """
    try:
        if not city_name.strip():
            raise InvalidParameterError("城市名称不能为空")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT origin_id, destination_id, distance, bus_duration, bus_cost,
            taxi_duration, taxi_cost, walk_duration, origin_type, destination_type, city_name
        FROM path_planning_in_city
        WHERE city_name = %s;
        """, (city_name.strip(),))

        intra_city_trans = cursor.fetchall()
        conn.close()

        if not intra_city_trans:
            raise DataNotFoundError(f"未找到城市'{city_name}'的市内交通数据")

        # 处理数据结构
        in_city_trans = {}
        for trans in intra_city_trans:
            key = f"{trans['origin_id']},{trans['destination_id']}"
            in_city_trans[key] = {
                'taxi_duration': trans['taxi_duration'],
                'taxi_cost': trans['taxi_cost'],
                'bus_duration': trans['bus_duration'],
                'bus_cost': trans['bus_cost']
            }

        return in_city_trans

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市内交通数据失败: {str(e)}")

@app.get("/poi", include_in_schema=False)
async def missing_poi_id(request: Request):
    raise InvalidParameterError("POI ID不能为空")
@app.get("/poi/{poi_id}",
         responses={
             200: {"description": "成功返回POI详细信息"},
             400: {"description": "POI ID参数错误", "model": ErrorResponse},
             404: {"description": "未找到POI信息", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
async def get_poi_by_id(
        poi_id: str = Path(..., description="POI的ID", min_length=1, max_length=50)
):
    """
    根据ID获取POI详细信息，自动判断POI类型

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "POI ID不能为空"}
    - 404: {"error": "Data Not Found", "message": "未找到ID为'XXX'的POI"}
    """
    try:
        if not poi_id.strip():
            raise InvalidParameterError("POI ID不能为空")


        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        # 首先尝试在景点表中查找
        # 获取景点数据
        cursor.execute("""
            SELECT attraction_id as id, attraction_name as name, avg_consumption as cost,
                          attraction_type as type, rating, suggested_duration as duration, city_name
            FROM poi_attraction
            WHERE attraction_id = %s
        """, (poi_id.strip(),))

        result = cursor.fetchone()

        # 如果景点表中未找到，则在酒店表中查找
        if not result:
            # 获取酒店数据
            cursor.execute("""
                SELECT accommodation_id as id, accommodation_name as name, avg_price as cost,
                              accommodation_type as type, rating, feature_hotel_type as feature, city_name
                FROM poi_accommodation
                WHERE accommodation_id = %s
            """, (poi_id.strip(),))
            result = cursor.fetchone()

        # 如果酒店表中未找到，则在餐厅表中查找
        if not result:
            cursor.execute("""
                SELECT restaurant_id as id, restaurant_name as name, avg_price as cost, rating,
                        restaurant_type as type, recommended_food, queue_time, consumption_time as duration, city_name
                FROM poi_restaurant
                WHERE restaurant_id = %s
            """, (poi_id.strip(),))
            result = cursor.fetchone()

        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到ID为'{poi_id}'的POI")

        # 处理时间类型数据
        if result['type'] == 'attraction':
            if result.get('open_time') and isinstance(result['open_time'], timedelta):
                seconds = int(result['open_time'].total_seconds())
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                result['open_time'] = f"{hours:02d}:{minutes:02d}"

            if result.get('close_time') and isinstance(result['close_time'], timedelta):
                seconds = int(result['close_time'].total_seconds())
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                result['close_time'] = f"{hours:02d}:{minutes:02d}"

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取POI信息失败: {str(e)}")


@app.post("/transport-params",
          response_model=Duration,
          responses={
              200: {"description": "成功返回交通参数"},
              400: {"description": "请求参数错误", "model": ErrorResponse},
              404: {"description": "未找到交通参数", "model": ErrorResponse},
              422: {"description": "参数验证失败", "model": ErrorResponse},
              503: {"description": "数据库连接失败", "model": ErrorResponse}
          })
def get_transport_params(params: DurationParams):
    """
    两点间交通参数获取

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "起点ID和终点ID不能相同"}
    - 404: {"error": "Data Not Found", "message": "未找到从'XXX'到'YYY'的交通参数"}
    - 422: {"error": "Validation Error", "message": "请求参数验证失败"}
    """
    try:
        if not params.origin_id or not params.destination_id:
            raise InvalidParameterError("起点ID和终点ID不能为空")

        if params.origin_id == params.destination_id:
            raise InvalidParameterError("起点ID和终点ID不能相同")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT bus_duration, bus_cost,
            taxi_duration, taxi_cost
        FROM path_planning_in_city
        WHERE origin_id = %s AND destination_id = %s;
        """, (params.origin_id, params.destination_id))

        intra_city_trans = cursor.fetchone()
        conn.close()

        if not intra_city_trans:
            raise DataNotFoundError(f"未找到从'{params.origin_id}'到'{params.destination_id}'的交通参数")

        return intra_city_trans

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取交通参数失败: {str(e)}")


@app.get("/train",
         response_model=TrainInfo,
         responses={
             200: {"description": "成功返回火车信息"},
             400: {"description": "请求参数错误", "model": ErrorResponse},
             404: {"description": "未找到火车信息", "model": ErrorResponse},
             503: {"description": "数据库连接失败", "model": ErrorResponse}
         })
async def get_train_by_number(
        train_number: str = Query(..., description="火车车次号", min_length=1, max_length=20),
        origin_id: str = Query(..., description="始发站ID", min_length=1, max_length=50),
        destination_id: str = Query(..., description="终点站ID", min_length=1, max_length=50)
):
    """
    根据车次号查询火车信息，包括始发站、始发城市、终点站、终点城市

    异常响应示例:
    - 400: {"error": "Invalid Parameter", "message": "车次号格式不正确"}
    - 404: {"error": "Data Not Found", "message": "未找到车次号为'XXX'的列车"}
    """
    try:
        if not train_number.strip() or not origin_id.strip() or not destination_id.strip():
            raise InvalidParameterError("车次号、始发站ID和终点站ID不能为空")

        if origin_id.strip() == destination_id.strip():
            raise InvalidParameterError("始发站ID和终点站ID不能相同")

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT train_number, origin_id, origin_city, origin_station, destination_id,
                   destination_city, destination_station, price, duration
            FROM train_info
            WHERE train_number = %s AND origin_id = %s AND destination_id = %s;
        """, (train_number.strip(), origin_id.strip(), destination_id.strip()))

        result = cursor.fetchone()
        conn.close()

        if not result:
            raise DataNotFoundError(f"未找到车次号为'{train_number}'的列车")

        return result

    except (DatabaseConnectionError, DataNotFoundError, InvalidParameterError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取火车信息失败: {str(e)}")


@app.get("/all-cities",
         response_model=List[City],
         responses={
             200: {"description": "成功返回所有城市数据"},
             503: {"description": "数据库连接失败", "model": ErrorResponse},
             500: {"description": "服务器内部错误", "model": ErrorResponse}
         })
async def get_all_cities():
    """
    获取系统中所有可用的城市数据

    返回城市的基本信息：城市ID、城市名称和所属省份

    异常响应示例:
    - 503: {"error": "Database Connection Error", "message": "数据库服务器无法连接"}
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        # 查询所有城市数据
        cursor.execute("""
            SELECT city_code, city_name
            FROM poi_city
            ORDER BY city_code, city_name
        """)

        cities = cursor.fetchall()
        conn.close()

        # 如果没有数据，返回空列表而不是报错
        if not cities:
            print("数据库中无城市数据")
            return []

        return cities

    except DatabaseConnectionError:
        raise
    except pymysql.Error as e:
        error_code = e.args[0] if e.args else 0
        raise HTTPException(
            status_code=500,
            detail=f"数据库查询城市列表错误 (错误代码: {error_code}): {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取城市数据失败: {str(e)}"
        )


    


    
def main():
    print('=======start=======')
    args = parse_parameters()
    uvicorn.run(
        app='baseConsultApi:app', 
        host="0.0.0.0", 
        port=int(args.api_server_port),
        reload=False)


if __name__ == "__main__":
    main()