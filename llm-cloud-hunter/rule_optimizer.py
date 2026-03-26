'''
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai.types.chat import ChatCompletion

from prompts.rule_optimizing_prompts import rule_optimizing_system_prompt, generate_rule_optimizing_user_prompt
from utils import validate_rule


class RuleOptimizer:
    def __init__(self, model_name: str = 'chatgpt-4o-latest', api_key: str = None, temperature: float = 0.5):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key if api_key else os.getenv('OPENAI_API_KEY'))
        self.temperature = temperature

    @staticmethod
    def _generate_optimization_messages(rule: dict) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": rule_optimizing_system_prompt},
            {"role": "user", "content": generate_rule_optimizing_user_prompt(rule)}
        ]

    def _send_optimization_request(self, messages: list[dict[str, str]]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
            response_format={"type": "json_object"}
        )

    def _optimize_rule(self, rule: dict) -> dict | None:
        messages = self._generate_optimization_messages(rule)

        try:
            response = self._send_optimization_request(messages)
        except Exception as e:
            logging.error(f"Error optimizing rule: {e}")
            return None

        response = response.choices[0].message.content
        rule = json.loads(response)
        rule = validate_rule(rule)

        return rule

    def optimize_rules(self, rules: dict | list[dict]) -> list[dict] | None:
        if isinstance(rules, dict):
            rules = [rules]

        results = [None] * len(rules)
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(self._optimize_rule, rule): index for index, rule in enumerate(rules)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                rule = future.result()
                results[index] = rule

        results = [result for result in results if result is not None]

        if results:
            return results
        return None
'''
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from openai.types.chat import ChatCompletion

from prompts.rule_optimizing_prompts import rule_optimizing_system_prompt, generate_rule_optimizing_user_prompt
from utils import validate_rule


class RuleOptimizer:
    def __init__(self, model_name: str = 'chatgpt-4o-latest', api_key: str = None, temperature: float = 0.5):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key if api_key else os.getenv('OPENAI_API_KEY'))
        self.temperature = temperature

    @staticmethod
    def _generate_optimization_messages(rule: dict) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": rule_optimizing_system_prompt},
            {"role": "user", "content": generate_rule_optimizing_user_prompt(rule)}
        ]

    def _send_optimization_request(self, messages: list[dict[str, str]]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
            response_format={"type": "json_object"}
        )

    def _optimize_rule(self, rule: dict) -> dict | None:
        # rule 자체가 이상하면 스킵
        if not isinstance(rule, dict) or not rule:
            logging.warning("Skip optimizing: invalid rule object.")
            return None

        messages = self._generate_optimization_messages(rule)

        try:
            response = self._send_optimization_request(messages)
            content = response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error optimizing rule (request failed): {e}")
            return None

        # JSON 파싱/검증 방어
        try:
            parsed = json.loads(content)
        except Exception as e:
            logging.error(f"Error optimizing rule (json parse failed): {e} | content={content[:300]}")
            return None

        try:
            parsed = validate_rule(parsed)
        except Exception as e:
            logging.error(f"Error optimizing rule (validate_rule failed): {e}")
            return None

        return parsed

    def optimize_rules(self, rules: dict | list[dict] | None) -> list[dict]:
        # ✅ 핵심: None/빈값 방어 (여기서 크래시 방지)
        if rules is None:
            logging.warning("optimize_rules called with None. Returning empty list.")
            return []

        # dict 하나면 리스트로
        if isinstance(rules, dict):
            rules = [rules]

        # 리스트가 아니거나, 빈 리스트면 빈 결과
        if not isinstance(rules, list) or len(rules) == 0:
            logging.warning(f"optimize_rules called with invalid/empty type: {type(rules)}. Returning empty list.")
            return []

        results = [None] * len(rules)

        # ThreadPool에서 예외가 올라올 수 있으니 future.result()도 try/except
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._optimize_rule, rule): index
                for index, rule in enumerate(rules)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    rule = future.result()
                except Exception as e:
                    logging.error(f"Error optimizing rule (future failed): {e}")
                    rule = None
                results[index] = rule

        # None 제거하고 리스트로 반환 (✅ 이제 None 대신 빈 리스트 반환)
        results = [result for result in results if result is not None]
        return results
