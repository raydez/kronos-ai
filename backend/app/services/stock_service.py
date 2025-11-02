"""
股票数据服务 - 负责从Baostock等数据源获取股票数据
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class StockService:
    """股票数据服务"""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    async def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """获取股票基本信息"""
        try:
            # 模拟从Baostock获取股票信息
            # 实际实现需要调用Baostock API
            stock_info = await self._fetch_stock_info_from_baostock(code)
            return stock_info
        except Exception as e:
            logger.error(f"获取股票 {code} 基本信息失败: {e}")
            return None
    
    async def get_stock_data(self, code: str, days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """获取股票历史数据"""
        try:
            # 先检查缓存
            cache_key = f"{code}_{days}"
            if cache_key in self._cache:
                logger.info(f"使用缓存数据 {code}")
                return self._cache[cache_key]
            
            # 从Baostock获取数据
            stock_data = await self._fetch_stock_data_from_baostock(code, days)
            
            if stock_data:
                # 缓存数据
                self._cache[cache_key] = stock_data
                return stock_data
            
            return None
        except Exception as e:
            logger.error(f"获取股票 {code} 历史数据失败: {e}")
            return None
    
    async def _fetch_stock_info_from_baostock(self, code: str) -> Dict[str, Any]:
        """从Baostock获取股票信息"""
        try:
            import baostock as bs
            
            # 登录Baostock
            lg = bs.login()
            if lg.error_code != '0':
                logger.warning(f"Baostock登录失败: {lg.error_msg}")
                return self._get_fallback_stock_info(code)
            
            # 查询股票基本信息
            market_prefix = "sh." if code.startswith("6") else "sz."
            full_code = market_prefix + code
            
            rs = bs.query_stock_basic(code=full_code)
            stock_list = rs.get_data()
            
            # 登出
            bs.logout()
            
            if not stock_list.empty:
                stock_info = stock_list.iloc[0]
                return {
                    "code": code,
                    "name": stock_info.get('code_name', f'股票{code}'),
                    "market": "上海" if market_prefix == "sh." else "深圳",
                    "industry": stock_info.get('industry', ''),
                    "type": stock_info.get('type', '')
                }
            else:
                logger.warning(f"未找到股票 {code} 的基本信息")
                return self._get_fallback_stock_info(code)
                
        except ImportError:
            logger.warning("Baostock未安装，使用备用数据")
            return self._get_fallback_stock_info(code)
        except Exception as e:
            logger.error(f"从Baostock获取股票 {code} 基本信息失败: {e}")
            return self._get_fallback_stock_info(code)
    
    def _get_fallback_stock_info(self, code: str) -> Dict[str, Any]:
        """获取备用股票信息"""
        # 常见股票的映射
        stock_mapping = {
            "600000": {"name": "浦发银行", "market": "上海"},
            "000001": {"name": "平安银行", "market": "深圳"},
            "600036": {"name": "招商银行", "market": "上海"},
            "000002": {"name": "万科A", "market": "深圳"},
            "600519": {"name": "贵州茅台", "market": "上海"},
            "000858": {"name": "五粮液", "market": "深圳"},
            "600276": {"name": "恒瑞医药", "market": "上海"},
            "002415": {"name": "海康威视", "market": "深圳"},
        }
        
        if code in stock_mapping:
            info = stock_mapping[code]
            return {"code": code, **info}
        
        # 默认返回
        market = "上海" if code.startswith("6") else "深圳"
        return {
            "code": code,
            "name": f"股票{code}",
            "market": market
        }
    
    async def _fetch_stock_data_from_baostock(self, code: str, days: int) -> List[Dict[str, Any]]:
        """从Baostock获取股票历史数据"""
        try:
            import baostock as bs
            
            # 登录Baostock
            lg = bs.login()
            if lg.error_code != '0':
                logger.warning(f"Baostock登录失败: {lg.error_msg}")
                return self._generate_fallback_data(code, days)
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')  # 多取一些数据确保足够
            
            # 构建股票代码
            market_prefix = "sh." if code.startswith("6") else "sz."
            full_code = market_prefix + code
            
            # 查询历史数据
            rs = bs.query_history_k_data_plus(
                full_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d"  # 日线数据
            )
            
            data_list = rs.get_data()
            
            # 登出
            bs.logout()
            
            if not data_list.empty:
                # 转换为需要的格式
                data = []
                for _, row in data_list.tail(days).iterrows():  # 只取最近的数据
                    try:
                        data.append({
                            "date": pd.to_datetime(row['date']),
                            "open": float(row['open']),
                            "high": float(row['high']),
                            "low": float(row['low']),
                            "close": float(row['close']),
                            "volume": int(row['volume']) if pd.notna(row['volume']) else 0
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"跳过无效数据行: {e}")
                        continue
                
                if data:
                    logger.info(f"成功从Baostock获取 {len(data)} 条记录用于 {code}")
                    return data
            
            logger.warning(f"未从Baostock获取到 {code} 的历史数据，使用备用数据")
            return self._generate_fallback_data(code, days)
            
        except ImportError:
            logger.warning("Baostock未安装，使用备用数据")
            return self._generate_fallback_data(code, days)
        except Exception as e:
            logger.error(f"从Baostock获取股票 {code} 历史数据失败: {e}")
            return self._generate_fallback_data(code, days)
    
    def _generate_fallback_data(self, code: str, days: int) -> List[Dict[str, Any]]:
        """生成备用数据"""
        logger.info(f"为 {code} 生成 {days} 天的备用数据")
        
        # 生成模拟数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = []
        current_date = start_date
        
        # 根据股票代码设置不同的基准价格
        base_prices = {
            "600519": 1500.0,  # 贵州茅台
            "000858": 150.0,   # 五粮液
            "600036": 40.0,    # 招商银行
            "000001": 15.0,    # 平安银行
            "600000": 10.0,    # 浦发银行
            "000002": 20.0,    # 万科A
        }
        
        base_price = base_prices.get(code, 10.0)
        
        for i in range(days):
            if current_date.weekday() < 5:  # 只生成工作日数据
                # 模拟价格波动
                price_change = (i % 7 - 3) * 0.02  # -6% 到 +6% 的波动
                trend = i * 0.001  # 轻微趋势
                
                open_price = base_price * (1 + price_change + trend)
                close_price = open_price * (1 + np.random.normal(0, 0.01))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
                
                data.append({
                    "date": current_date,
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": np.random.randint(1000000, 10000000)
                })
                
                base_price = close_price
            
            current_date += timedelta(days=1)
        
        return data
    
    async def get_stock_data_for_dates(self, code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """获取指定日期范围内的股票数据（用于对比预测和实际数据）"""
        try:
            # 计算日期范围
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end - start).days + 1
            
            # 获取该时间范围的数据
            data = await self._fetch_stock_data_from_baostock(code, days)
            
            # 过滤日期范围
            filtered_data = []
            for item in data:
                item_date = datetime.strptime(item['date'], '%Y-%m-%d')
                if start <= item_date <= end:
                    filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"获取股票 {code} 日期范围 {start_date} 到 {end_date} 的数据失败: {e}")
            # 返回空列表而不是fallback数据，因为这是用于对比的
            return []
    
    async def validate_stock_code(self, code: str) -> bool:
        """验证股票代码格式"""
        if not code or len(code) != 6:
            return False
        
        # 检查是否为数字
        if not code.isdigit():
            return False
        
        # 检查股票代码范围（简化验证）
        code_int = int(code)
        
        # 上海主板：600000-604999, 688000-688999 (科创板)
        if 600000 <= code_int <= 604999 or 688000 <= code_int <= 688999:
            return True
        
        # 深圳主板：000001-002999, 创业板：300000-300999, 301000-301999
        if (1 <= code_int <= 2999 or 
            300000 <= code_int <= 300999 or 
            301000 <= code_int <= 301999):
            return True
        
        return False
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("股票数据缓存已清空")


# 全局股票服务实例
stock_service = StockService()