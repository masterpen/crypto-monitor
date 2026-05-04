#!/bin/bash
# 量化交易系统部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_message "检查依赖..."
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        exit 1
    fi
    
    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装"
        exit 1
    fi
    
    print_message "依赖检查完成"
}

# 检查环境变量
check_env() {
    print_message "检查环境变量..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            print_warning ".env 文件不存在，正在从 .env.example 复制..."
            cp .env.example .env
            print_warning "请编辑 .env 文件并填写实际值"
            exit 1
        else
            print_error ".env 文件不存在"
            exit 1
        fi
    fi
    
    # 检查必要的环境变量
    source .env
    
    if [ -z "$BINANCE_API_KEY" ] || [ "$BINANCE_API_KEY" = "your_api_key_here" ]; then
        print_warning "BINANCE_API_KEY 未设置或使用默认值"
    fi
    
    if [ -z "$BINANCE_API_SECRET" ] || [ "$BINANCE_API_SECRET" = "your_api_secret_here" ]; then
        print_warning "BINANCE_API_SECRET 未设置或使用默认值"
    fi
    
    if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "your_postgres_password_here" ]; then
        print_warning "POSTGRES_PASSWORD 未设置或使用默认值"
    fi
    
    print_message "环境变量检查完成"
}

# 创建目录
create_directories() {
    print_message "创建目录..."
    
    mkdir -p data/cache
    mkdir -p data/storage
    mkdir -p logs
    mkdir -p reports
    
    print_message "目录创建完成"
}

# 构建镜像
build_images() {
    print_message "构建 Docker 镜像..."
    
    docker-compose build
    
    print_message "镜像构建完成"
}

# 启动服务
start_services() {
    print_message "启动服务..."
    
    docker-compose up -d
    
    print_message "服务启动完成"
}

# 停止服务
stop_services() {
    print_message "停止服务..."
    
    docker-compose down
    
    print_message "服务停止完成"
}

# 重启服务
restart_services() {
    print_message "重启服务..."
    
    docker-compose restart
    
    print_message "服务重启完成"
}

# 查看日志
view_logs() {
    local service=$1
    
    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

# 查看状态
view_status() {
    print_message "服务状态:"
    docker-compose ps
}

# 清理资源
cleanup() {
    print_message "清理资源..."
    
    docker-compose down -v
    docker system prune -f
    
    print_message "资源清理完成"
}

# 备份数据
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    
    print_message "备份数据到 $backup_dir..."
    
    mkdir -p "$backup_dir"
    
    # 备份配置
    cp -r config "$backup_dir/"
    
    # 备份数据
    if [ -d "data" ]; then
        cp -r data "$backup_dir/"
    fi
    
    # 备份日志
    if [ -d "logs" ]; then
        cp -r logs "$backup_dir/"
    fi
    
    # 备份报告
    if [ -d "reports" ]; then
        cp -r reports "$backup_dir/"
    fi
    
    print_message "数据备份完成"
}

# 恢复数据
restore_data() {
    local backup_dir=$1
    
    if [ -z "$backup_dir" ]; then
        print_error "请指定备份目录"
        exit 1
    fi
    
    if [ ! -d "$backup_dir" ]; then
        print_error "备份目录不存在: $backup_dir"
        exit 1
    fi
    
    print_message "从 $backup_dir 恢复数据..."
    
    # 恢复配置
    if [ -d "$backup_dir/config" ]; then
        cp -r "$backup_dir/config" .
    fi
    
    # 恢复数据
    if [ -d "$backup_dir/data" ]; then
        cp -r "$backup_dir/data" .
    fi
    
    # 恢复报告
    if [ -d "$backup_dir/reports" ]; then
        cp -r "$backup_dir/reports" .
    fi
    
    print_message "数据恢复完成"
}

# 显示帮助
show_help() {
    echo "量化交易系统部署脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build       构建 Docker 镜像"
    echo "  start       启动所有服务"
    echo "  stop        停止所有服务"
    echo "  restart     重启所有服务"
    echo "  status      查看服务状态"
    echo "  logs        查看日志（可指定服务名）"
    echo "  cleanup     清理资源"
    echo "  backup      备份数据"
    echo "  restore     恢复数据（需要指定备份目录）"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 logs trading"
    echo "  $0 restore backups/20240101_120000"
}

# 主函数
main() {
    local command=$1
    shift
    
    # 检查依赖
    check_dependencies
    
    # 检查环境变量
    check_env
    
    # 创建目录
    create_directories
    
    case $command in
        build)
            build_images
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            view_status
            ;;
        logs)
            view_logs "$@"
            ;;
        cleanup)
            cleanup
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"