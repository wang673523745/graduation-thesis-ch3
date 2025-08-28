#!/usr/bin/env python3
"""
红外可见光Agent系统 - 测试运行脚本
可以分别运行不同模块的测试
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_pattern=None, module=None, verbose=False, coverage=False):
    """运行测试"""
    # 获取测试目录
    test_dir = Path(__file__).parent
    
    # 构建pytest命令
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # 添加测试目录
    cmd.append(str(test_dir))
    
    # 添加测试模式
    if test_pattern:
        cmd.append(f"-k {test_pattern}")
    
    if module:
        cmd.append(f"test_{module}.py")
    
    # 运行测试
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return result.returncode

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行红外可见光Agent系统测试")
    parser.add_argument("--module", "-m", choices=["preprocess", "fusion", "segment", "metrics", "report", "integration"], 
                       help="指定要测试的模块")
    parser.add_argument("--pattern", "-k", help="指定测试模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--coverage", "-c", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--all", "-a", action="store_true", help="运行所有测试")
    
    args = parser.parse_args()
    
    if args.all:
        # 运行所有测试
        print("运行所有测试...")
        return run_tests(verbose=args.verbose, coverage=args.coverage)
    
    elif args.module:
        # 运行指定模块的测试
        print(f"运行 {args.module} 模块测试...")
        return run_tests(module=args.module, verbose=args.verbose, coverage=args.coverage)
    
    elif args.pattern:
        # 运行匹配模式的测试
        print(f"运行匹配 '{args.pattern}' 的测试...")
        return run_tests(test_pattern=args.pattern, verbose=args.verbose, coverage=args.coverage)
    
    else:
        # 默认运行所有测试
        print("运行所有测试...")
        return run_tests(verbose=args.verbose, coverage=args.coverage)

if __name__ == "__main__":
    sys.exit(main())
