# 运行mysql数据库交互服务
# host->mysql数据库主机ip
# user->mysql数据库用户名
# password->mysql数据库密码
# port->mysql数据库端口
# database->mysql数据库名
# api_server_port->提供mysql数据库交互服务的端口
python server/baseConsultApi.py \
    --host  \
    --user  \
    --password  \
    --port  \
    --database  \
    --api_server_port 12457