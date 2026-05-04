# 量化交易系统 Makefile

.PHONY: help build start stop restart status logs clean backup restore test lint format

# 默认目标
help:
	@echo "量化交易系统"
	@echo ""
	@echo "可用命令:"
	@echo "  make build       构建 Docker 镜像"
	@echo "  make start       启动所有服务"
	@echo "  make stop        停止所有服务"
	@echo "  make restart     重启所有服务"
	@echo "  make status      查看服务状态"
	@echo "  make logs        查看日志"
	@echo "  make logs-trading 查看交易服务日志"
	@echo "  make logs-dashboard 查看 Dashboard 日志"
	@echo "  make clean       清理资源"
	@echo "  make backup      备份数据"
	@echo "  make restore     恢复数据（需要指定 BACKUP_DIR）"
	@echo "  make test        运行测试"
	@echo "  make lint        代码检查"
	@echo "  make format      代码格式化"
	@echo "  make install     安装依赖"
	@echo "  make run-backtest 运行回测"
	@echo "  make run-trading  运行交易"
	@echo "  make run-dashboard 运行 Dashboard"

# 构建 Docker 镜像
build:
	docker-compose build

# 启动所有服务
start:
	docker-compose up -d

# 停止所有服务
stop:
	docker-compose down

# 重启所有服务
restart:
	docker-compose restart

# 查看服务状态
status:
	docker-compose ps

# 查看日志
logs:
	docker-compose logs -f

# 查看交易服务日志
logs-trading:
	docker-compose logs -f trading

# 查看 Dashboard 日志
logs-dashboard:
	docker-compose logs -f dashboard

# 清理资源
clean:
	docker-compose down -v
	docker system prune -f

# 备份数据
backup:
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r config backups/$(shell date +%Y%m%d_%H%M%S)/
	@cp -r data backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@cp -r logs backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@cp -r reports backups/$(shell date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
	@echo "备份完成"

# 恢复数据
restore:
ifndef BACKUP_DIR
	@echo "请指定备份目录: make restore BACKUP_DIR=backups/20240101_120000"
	@exit 1
endif
	@cp -r $(BACKUP_DIR)/config . 2>/dev/null || true
	@cp -r $(BACKUP_DIR)/data . 2>/dev/null || true
	@cp -r $(BACKUP_DIR)/reports . 2>/dev/null || true
	@echo "恢复完成"

# 运行测试
test:
	python -m pytest tests/ -v --cov=src --cov-report=html

# 代码检查
lint:
	python -m flake8 src/ --max-line-length=120 --ignore=E501,W503
	python -m pylint src/ --disable=C0114,C0115,C0116

# 代码格式化
format:
	python -m black src/ --line-length=120
	python -m isort src/ --profile=black

# 安装依赖
install:
	pip install -r requirements.txt

# 运行回测
run-backtest:
	python run_backtest.py --symbol BTCUSDT --strategy TrendStrategy --days 365

# 运行交易
run-trading:
	python run_trading.py --symbol BTCUSDT --strategy TrendStrategy

# 运行 Dashboard
run-dashboard:
	streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0

# 安装开发依赖
install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 pylint black isort

# 生成覆盖率报告
coverage:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "覆盖率报告已生成: htmlcov/index.html"

# 检查代码质量
check: lint test
	@echo "代码质量检查完成"

# 清理 Python 缓存
clean-python:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "htmlcov" -delete
	find . -type f -name ".coverage" -delete

# 清理所有
clean-all: clean clean-python
	@echo "清理完成"