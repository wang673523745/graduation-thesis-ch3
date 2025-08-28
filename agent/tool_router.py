"""
红外可见光Agent系统 - 工具路由器
负责工具调用、重试机制、异常处理和性能监控
"""

import time
import logging
import traceback
from functools import wraps
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    status: str  # "ok", "fail", "timeout"
    latency: float
    output: Any
    error: Optional[str] = None
    retry_count: int = 0

def tool(name: str, max_retries: int = 3, timeout: float = 30.0):
    """工具装饰器，提供统一的异常处理和性能监控"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult:
            start_time = time.time()
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # 设置超时
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"工具 {name} 执行超时")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                    
                    # 执行工具
                    output = func(*args, **kwargs)
                    
                    # 取消超时
                    signal.alarm(0)
                    
                    latency = time.time() - start_time
                    return ToolResult(
                        tool_name=name,
                        status="ok",
                        latency=latency,
                        output=output,
                        retry_count=retry_count
                    )
                    
                except TimeoutError as e:
                    latency = time.time() - start_time
                    logger.warning(f"工具 {name} 超时 (尝试 {retry_count + 1}/{max_retries + 1})")
                    if retry_count == max_retries:
                        return ToolResult(
                            tool_name=name,
                            status="timeout",
                            latency=latency,
                            output=None,
                            error=str(e),
                            retry_count=retry_count
                        )
                        
                except Exception as e:
                    latency = time.time() - start_time
                    logger.error(f"工具 {name} 执行失败: {str(e)}")
                    if retry_count == max_retries:
                        return ToolResult(
                            tool_name=name,
                            status="fail",
                            latency=latency,
                            output=None,
                            error=str(e),
                            retry_count=retry_count
                        )
                
                retry_count += 1
                if retry_count <= max_retries:
                    time.sleep(1)  # 重试前等待1秒
            
            # 不应该到达这里
            return ToolResult(
                tool_name=name,
                status="fail",
                latency=time.time() - start_time,
                output=None,
                error="未知错误",
                retry_count=retry_count
            )
        
        return wrapper
    return decorator

class ToolRouter:
    """工具路由器，管理所有工具的注册和调用"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        self.execution_history: list = []
    
    def register_tool(self, name: str, func: Callable, config: Optional[Dict[str, Any]] = None):
        """注册工具"""
        self.tools[name] = func
        self.tool_configs[name] = config or {}
        logger.info(f"注册工具: {name}")
    
    def call(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """调用工具"""
        if tool_name not in self.tools:
            error_msg = f"工具 {tool_name} 未注册"
            logger.error(error_msg)
            return ToolResult(
                tool_name=tool_name,
                status="fail",
                latency=0.0,
                output=None,
                error=error_msg
            )
        
        # 记录执行开始
        execution_record = {
            "tool": tool_name,
            "params": params,
            "start_time": time.time(),
            "status": "running"
        }
        self.execution_history.append(execution_record)
        
        try:
            # 获取工具配置
            config = self.tool_configs.get(tool_name, {})
            max_retries = config.get("max_retries", 3)
            timeout = config.get("timeout", 30.0)
            
            # 调用工具
            result = self.tools[tool_name](**params)
            
            # 更新执行记录
            execution_record["status"] = "completed"
            execution_record["end_time"] = time.time()
            execution_record["latency"] = execution_record["end_time"] - execution_record["start_time"]
            
            return result
            
        except Exception as e:
            # 更新执行记录
            execution_record["status"] = "failed"
            execution_record["end_time"] = time.time()
            execution_record["error"] = str(e)
            
            logger.error(f"工具 {tool_name} 调用失败: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                status="fail",
                latency=time.time() - execution_record["start_time"],
                output=None,
                error=str(e)
            )
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息"""
        if tool_name not in self.tools:
            return None
        
        func = self.tools[tool_name]
        config = self.tool_configs.get(tool_name, {})
        
        return {
            "name": tool_name,
            "function": func.__name__,
            "doc": func.__doc__,
            "config": config
        }
    
    def list_tools(self) -> list:
        """列出所有注册的工具"""
        return list(self.tools.keys())
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        if not self.execution_history:
            return {"total_calls": 0, "success_rate": 0.0, "avg_latency": 0.0}
        
        total_calls = len(self.execution_history)
        successful_calls = len([r for r in self.execution_history if r.get("status") == "completed"])
        success_rate = successful_calls / total_calls if total_calls > 0 else 0.0
        
        latencies = [r.get("latency", 0) for r in self.execution_history if r.get("latency")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "recent_calls": self.execution_history[-10:]  # 最近10次调用
        }

# 全局工具路由器实例
router = ToolRouter()

def demo_tool_router():
    """演示工具路由器功能"""
    
    # 模拟工具函数
    @tool("test_tool", max_retries=2, timeout=5.0)
    def test_tool(param1: str, param2: int) -> Dict[str, Any]:
        """测试工具"""
        time.sleep(0.1)  # 模拟处理时间
        return {"result": f"处理 {param1} 和 {param2}", "status": "success"}
    
    @tool("failing_tool", max_retries=1, timeout=3.0)
    def failing_tool() -> Dict[str, Any]:
        """总是失败的工具"""
        raise ValueError("模拟错误")
    
    # 注册工具
    router.register_tool("test_tool", test_tool, {"max_retries": 2, "timeout": 5.0})
    router.register_tool("failing_tool", failing_tool, {"max_retries": 1, "timeout": 3.0})
    
    # 测试工具调用
    print("=== 工具路由器演示 ===")
    
    # 成功调用
    result1 = router.call("test_tool", {"param1": "hello", "param2": 42})
    print(f"成功调用结果: {result1}")
    
    # 失败调用
    result2 = router.call("failing_tool", {})
    print(f"失败调用结果: {result2}")
    
    # 获取统计信息
    stats = router.get_execution_stats()
    print(f"执行统计: {stats}")
    
    # 列出工具
    tools = router.list_tools()
    print(f"注册的工具: {tools}")

if __name__ == "__main__":
    demo_tool_router()
