SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for city_info
-- ----------------------------
DROP TABLE IF EXISTS `city_info`;
CREATE TABLE `city_info`  (
  `city_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `up_ad_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '省级区域名称',
  `up_ad_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '省级区域代码',
  `cityname` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `adcode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '具体区域代码',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '市级区域代码',
  PRIMARY KEY (`city_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for path_planning_cross_city
-- ----------------------------
DROP TABLE IF EXISTS `path_planning_cross_city`;
CREATE TABLE `path_planning_cross_city`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联交通表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联交通表） 关联poi_master.poi_id',
  `train_plan` json NULL COMMENT '火车方案',
  `airplane_plan` json NULL COMMENT '飞机方案',
  `origin_type` enum('火车站','机场','汽车站') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `destination_type` enum('火车站','机场','汽车站') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `path_planning_cross_city_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `path_planning_cross_city_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '跨市路径规划表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for path_planning_in_city
-- ----------------------------
DROP TABLE IF EXISTS `path_planning_in_city`;
CREATE TABLE `path_planning_in_city`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联起点/住宿/交通/美食/景点表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联起点/住宿/交通/美食/景点表） 关联poi_master.poi_id',
  `distance` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '距离(m)',
  `taxi_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '打车时间花费(min)',
  `taxi_cost` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '打车花费(rmb)',
  `bus_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '公交时间花费',
  `bus_cost` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '公交花费',
  `walk_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '步行时间花费',
  `origin_type` enum('起点','景点','住宿','饭店') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `destination_type` enum('起点','景点','住宿','饭店') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `path_planning_in_city_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `path_planning_in_city_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '市内路径规划表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_accommodation
-- ----------------------------
DROP TABLE IF EXISTS `poi_accommodation`;
CREATE TABLE `poi_accommodation`  (
  `accommodation_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `avg_price` decimal(8, 2) NULL DEFAULT NULL COMMENT '平均价格',
  `rating` decimal(3, 2) NULL DEFAULT NULL,
  `feature_hotel_type` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '酒店特色',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`accommodation_id`) USING BTREE,
  CONSTRAINT `poi_accommodation_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_attraction
-- ----------------------------
DROP TABLE IF EXISTS `poi_attraction`;
CREATE TABLE `poi_attraction`  (
  `attraction_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `attraction_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '景点名称',
  `attraction_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '详细地址',
  `avg_consumption` decimal(10, 2) NULL DEFAULT NULL COMMENT '平均消费（单位：元）',
  `attraction_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '景点类型',
  `rating` decimal(3, 2) NULL DEFAULT NULL COMMENT '评分（0-5分）',
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `open_time` time NULL DEFAULT NULL COMMENT '开店时间',
  `close_time` time NULL DEFAULT NULL COMMENT '关店时间',
  `suggested_duration` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '建立游玩时长(单位分钟)',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`attraction_id`) USING BTREE,
  CONSTRAINT `poi_attraction_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_city
-- ----------------------------
DROP TABLE IF EXISTS `poi_city`;
CREATE TABLE `poi_city`  (
  `city_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '城市代码',
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '城市名称',
  PRIMARY KEY (`city_code`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for poi_master
-- ----------------------------
DROP TABLE IF EXISTS `poi_master`;
CREATE TABLE `poi_master`  (
  `poi_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `poi_type` enum('starting','accommodation','transport','restaurant','attraction') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`poi_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = 'POI统一入口表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_restaurant
-- ----------------------------
DROP TABLE IF EXISTS `poi_restaurant`;
CREATE TABLE `poi_restaurant`  (
  `restaurant_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_type` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `avg_price` decimal(8, 2) NULL DEFAULT NULL COMMENT '人均消费',
  `rating` decimal(3, 2) NULL DEFAULT NULL,
  `business_hours` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '营业时间段',
  `recommended_food` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '推荐菜品',
  `queue_time` int NULL DEFAULT NULL COMMENT '平均排队时间（分钟）',
  `consumption_time` int NULL DEFAULT NULL COMMENT '平均消费时间（分钟）',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`restaurant_id`) USING BTREE,
  CONSTRAINT `poi_restaurant_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_starting_point
-- ----------------------------
DROP TABLE IF EXISTS `poi_starting_point`;
CREATE TABLE `poi_starting_point`  (
  `starting_point_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `point_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `point_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`starting_point_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_transport
-- ----------------------------
DROP TABLE IF EXISTS `poi_transport`;
CREATE TABLE `poi_transport`  (
  `transport_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `transport_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '站点名称',
  `transport_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `transport_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `open_time` time NULL DEFAULT NULL COMMENT '开放时间',
  `close_time` time NULL DEFAULT NULL COMMENT '关闭时间',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `distance_type` enum('长途','短途') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`transport_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_type
-- ----------------------------
DROP TABLE IF EXISTS `poi_type`;
CREATE TABLE `poi_type`  (
  `type_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `type_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `big_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `mid_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `sub_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`type_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for train_info
-- ----------------------------
DROP TABLE IF EXISTS `train_info`;
CREATE TABLE `train_info`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联交通表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联交通表） 关联poi_master.poi_id',
  `train_number` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '车次',
  `duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '乘坐时间',
  `price` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '票价',
  `origin_station` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发站',
  `origin_city` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发城市',
  `origin_city_code` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发城市',
  `destination_station` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的站',
  `destination_city` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的城市',
  `destination_city_code` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的城市',
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `train_info_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `train_info_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '火车信息表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Triggers structure for table poi_accommodation
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_accommodation`;
delimiter ;;
CREATE TRIGGER `trig_accommodation` AFTER INSERT ON `poi_accommodation` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.accommodation_id, 'accommodation');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_attraction
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_attraction`;
delimiter ;;
CREATE TRIGGER `trig_attraction` AFTER INSERT ON `poi_attraction` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.attraction_id, 'attraction');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_restaurant
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_restaurant`;
delimiter ;;
CREATE TRIGGER `trig_restaurant` AFTER INSERT ON `poi_restaurant` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.restaurant_id, 'restaurant');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_starting_point
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_starting_point`;
delimiter ;;
CREATE TRIGGER `trig_starting_point` AFTER INSERT ON `poi_starting_point` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.starting_point_id, 'starting');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_transport
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_transport`;
delimiter ;;
CREATE TRIGGER `trig_transport` AFTER INSERT ON `poi_transport` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.transport_id, 'transport');
END
;;
delimiter ;

SET FOREIGN_KEY_CHECKS = 1;
