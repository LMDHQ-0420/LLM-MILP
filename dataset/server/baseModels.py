# 数据模型定义
from typing import Optional, Any, Dict, List

from pydantic import BaseModel, field_validator


class CrossCityTransport(BaseModel):
    origin_id: str
    destination_id: str
    train_number: str
    duration: str
    cost: str
    origin_station: str
    destination_station: str


# noinspection PyNestedDecorators
class Attraction(BaseModel):
    id: str
    name: str
    cost: float
    type: str
    rating: float
    duration: float


class Accommodation(BaseModel):
    id: str
    name: str
    cost: float
    type: str
    rating: float
    feature: str


class Restaurant(BaseModel):
    id: str
    name: str
    cost: float
    type: str
    duration: Optional[float]
    rating: float
    recommended_food: str
    duration: Optional[float]
    queue_time: Optional[float]


class Duration(BaseModel):
    bus_duration: float
    bus_cost: float
    taxi_duration: float
    taxi_cost: float


class DurationParams(BaseModel):
    origin_id: str
    destination_id: str


class TrainInfo(BaseModel):
    train_number: str
    origin_id: str
    origin_city: str
    origin_station: str
    destination_id: str
    destination_city: str
    destination_station: str
    price: float | str
    duration: float | str

class City(BaseModel):
    city_code: str
    city_name: str




class EvaluationResult(BaseModel):
    """评估结果数据模型，包含评估的详细信息"""
    avg_score: Dict[str, Any]  # 平均分数
    samples: List[Dict[str, Any]]  # 各个样本的分数列表
    total_time: float  # 总耗时(秒)
    contestant_id: str  # 选手唯一标识


class ApiResponse(BaseModel):
    """API通用响应体模型"""
    code: int  # 响应状态码：200表示成功，其他为错误码
    message: str  # 响应信息描述
    data: Optional[EvaluationResult] = None  # 成功时返回的数据，失败时为None
