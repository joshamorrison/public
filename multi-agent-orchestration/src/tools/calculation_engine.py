"""
Calculation Engine Tool

Provides mathematical and analytical calculation capabilities for agents.
"""

import asyncio
import math
import statistics
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json


class CalculationEngineTool:
    """
    Calculation engine that provides mathematical and analytical capabilities.
    
    Features:
    - Basic mathematical operations
    - Statistical analysis
    - Financial calculations
    - Data analysis and metrics
    - Expression evaluation (safe)
    """
    
    def __init__(self):
        """Initialize calculation engine."""
        self.math_functions = {
            'abs', 'ceil', 'floor', 'round', 'sqrt', 'pow', 'log', 'log10',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'degrees', 'radians',
            'pi', 'e'
        }
    
    async def calculate(self, operation: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform calculation operation on provided data.
        
        Args:
            operation: Type of calculation (math, stats, financial, etc.)
            data: Input data for calculation
            **kwargs: Additional parameters for specific operations
            
        Returns:
            Calculation results with metadata
        """
        start_time = datetime.now()
        
        try:
            if operation == "basic_math":
                result = await self._basic_math(data, **kwargs)
            elif operation == "statistics":
                result = await self._calculate_statistics(data, **kwargs)
            elif operation == "financial":
                result = await self._financial_calculations(data, **kwargs)
            elif operation == "data_analysis":
                result = await self._analyze_data(data, **kwargs)
            elif operation == "expression":
                result = await self._evaluate_expression(data, **kwargs)
            else:
                result = {"error": f"Unknown operation: {operation}"}
            
            # Add calculation metadata
            result.update({
                "operation": operation,
                "calculation_time": (datetime.now() - start_time).total_seconds(),
                "calculated_at": datetime.now().isoformat(),
                "tool": "CalculationEngineTool"
            })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation,
                "calculated_at": datetime.now().isoformat()
            }
    
    async def _basic_math(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform basic mathematical operations."""
        await asyncio.sleep(0.05)  # Simulate processing
        
        operation_type = data.get("type", "arithmetic")
        
        if operation_type == "arithmetic":
            a = data.get("a", 0)
            b = data.get("b", 0)
            operator = data.get("operator", "+")
            
            operations = {
                "+": lambda x, y: x + y,
                "-": lambda x, y: x - y,
                "*": lambda x, y: x * y,
                "/": lambda x, y: x / y if y != 0 else float('inf'),
                "//": lambda x, y: x // y if y != 0 else float('inf'),
                "%": lambda x, y: x % y if y != 0 else 0,
                "**": lambda x, y: x ** y,
                "^": lambda x, y: x ** y
            }
            
            if operator in operations:
                result_value = operations[operator](a, b)
                return {
                    "result": result_value,
                    "expression": f"{a} {operator} {b}",
                    "operands": [a, b],
                    "operator": operator
                }
            else:
                return {"error": f"Unknown operator: {operator}"}
        
        elif operation_type == "advanced":
            value = data.get("value", 0)
            function = data.get("function", "sqrt")
            
            advanced_ops = {
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "abs": abs,
                "ceil": math.ceil,
                "floor": math.floor
            }
            
            if function in advanced_ops:
                result_value = advanced_ops[function](value)
                return {
                    "result": result_value,
                    "function": function,
                    "input": value
                }
            else:
                return {"error": f"Unknown function: {function}"}
    
    async def _calculate_statistics(self, data: List[Union[int, float]], **kwargs) -> Dict[str, Any]:
        """Calculate statistical measures for a dataset."""
        await asyncio.sleep(0.1)
        
        if not data or not isinstance(data, list):
            return {"error": "Data must be a non-empty list of numbers"}
        
        try:
            # Convert to numbers
            numbers = [float(x) for x in data if isinstance(x, (int, float)) or str(x).replace('.', '').replace('-', '').isdigit()]
            
            if not numbers:
                return {"error": "No valid numbers found in data"}
            
            # Basic statistics
            mean_val = statistics.mean(numbers)
            median_val = statistics.median(numbers)
            
            # Handle mode calculation
            try:
                mode_val = statistics.mode(numbers)
            except statistics.StatisticsError:
                mode_val = None  # No unique mode
            
            # Standard deviation (handle single value case)
            if len(numbers) > 1:
                std_dev = statistics.stdev(numbers)
                variance = statistics.variance(numbers)
            else:
                std_dev = 0
                variance = 0
            
            # Range and quartiles
            sorted_nums = sorted(numbers)
            q1 = statistics.median(sorted_nums[:len(sorted_nums)//2])
            q3 = statistics.median(sorted_nums[(len(sorted_nums)+1)//2:])
            
            return {
                "count": len(numbers),
                "sum": sum(numbers),
                "mean": round(mean_val, 4),
                "median": median_val,
                "mode": mode_val,
                "std_deviation": round(std_dev, 4),
                "variance": round(variance, 4),
                "min": min(numbers),
                "max": max(numbers),
                "range": max(numbers) - min(numbers),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "original_data_length": len(data),
                "valid_numbers": len(numbers)
            }
            
        except Exception as e:
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    async def _financial_calculations(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform financial calculations."""
        await asyncio.sleep(0.1)
        
        calc_type = data.get("type", "compound_interest")
        
        if calc_type == "compound_interest":
            principal = data.get("principal", 1000)
            rate = data.get("rate", 0.05)  # 5% annual rate
            time = data.get("time", 1)  # 1 year
            compound_frequency = data.get("compound_frequency", 1)  # Annually
            
            amount = principal * (1 + rate/compound_frequency) ** (compound_frequency * time)
            interest = amount - principal
            
            return {
                "principal": principal,
                "rate": rate,
                "time": time,
                "compound_frequency": compound_frequency,
                "final_amount": round(amount, 2),
                "interest_earned": round(interest, 2),
                "roi_percentage": round((interest/principal) * 100, 2)
            }
        
        elif calc_type == "present_value":
            future_value = data.get("future_value", 1000)
            rate = data.get("rate", 0.05)
            time = data.get("time", 1)
            
            present_value = future_value / (1 + rate) ** time
            
            return {
                "future_value": future_value,
                "discount_rate": rate,
                "time": time,
                "present_value": round(present_value, 2),
                "discount_amount": round(future_value - present_value, 2)
            }
        
        elif calc_type == "loan_payment":
            principal = data.get("principal", 100000)
            annual_rate = data.get("annual_rate", 0.05)
            years = data.get("years", 30)
            
            monthly_rate = annual_rate / 12
            num_payments = years * 12
            
            if monthly_rate > 0:
                monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = principal / num_payments
            
            total_payment = monthly_payment * num_payments
            total_interest = total_payment - principal
            
            return {
                "loan_amount": principal,
                "annual_rate": annual_rate,
                "loan_term_years": years,
                "monthly_payment": round(monthly_payment, 2),
                "total_payments": round(total_payment, 2),
                "total_interest": round(total_interest, 2),
                "number_of_payments": num_payments
            }
        
        else:
            return {"error": f"Unknown financial calculation type: {calc_type}"}
    
    async def _analyze_data(self, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Analyze structured data for insights."""
        await asyncio.sleep(0.2)
        
        if not data or not isinstance(data, list):
            return {"error": "Data must be a non-empty list"}
        
        analysis_type = kwargs.get("analysis_type", "summary")
        
        if analysis_type == "summary":
            # Basic data summary
            total_records = len(data)
            
            # Find numeric columns
            numeric_columns = set()
            for record in data:
                if isinstance(record, dict):
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            numeric_columns.add(key)
            
            # Calculate summaries for numeric columns
            column_summaries = {}
            for column in numeric_columns:
                values = [record.get(column, 0) for record in data if isinstance(record, dict) and column in record]
                if values:
                    column_summaries[column] = {
                        "count": len(values),
                        "sum": sum(values),
                        "average": round(sum(values) / len(values), 2),
                        "min": min(values),
                        "max": max(values)
                    }
            
            return {
                "total_records": total_records,
                "numeric_columns": list(numeric_columns),
                "column_summaries": column_summaries,
                "analysis_type": analysis_type
            }
        
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    async def _evaluate_expression(self, expression: str, **kwargs) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions."""
        await asyncio.sleep(0.05)
        
        # Security: Only allow safe mathematical expressions
        allowed_chars = set('0123456789+-*/()., ')
        allowed_words = {'abs', 'round', 'max', 'min', 'sum'}
        
        # Basic validation
        if not all(c in allowed_chars or c.isalnum() for c in expression):
            return {"error": "Expression contains invalid characters"}
        
        # Check for allowed functions only
        words_in_expr = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression)
        if not all(word in allowed_words for word in words_in_expr):
            return {"error": "Expression contains disallowed functions"}
        
        try:
            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "sum": sum
            }
            
            result = eval(expression, safe_dict)
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
            
        except Exception as e:
            return {"error": f"Expression evaluation failed: {str(e)}"}


# Standalone async functions for simple usage
async def calculate_stats(numbers: List[Union[int, float]]) -> Dict[str, Any]:
    """Simple statistics calculation function for agent use."""
    engine = CalculationEngineTool()
    return await engine.calculate("statistics", numbers)

async def evaluate_math(expression: str) -> Dict[str, Any]:
    """Simple math expression evaluation for agent use."""
    engine = CalculationEngineTool()
    return await engine.calculate("expression", expression)

async def compound_interest(principal: float, rate: float, time: float) -> Dict[str, Any]:
    """Simple compound interest calculation for agent use."""
    engine = CalculationEngineTool()
    data = {"type": "compound_interest", "principal": principal, "rate": rate, "time": time}
    return await engine.calculate("financial", data)