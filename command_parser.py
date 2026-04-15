import re

class CommandParser:
    """
    轻量级自然语言指令解析器，支持同义词和数量词映射。
    """
    def __init__(self):
        # 动作关键词映射
        self.action_map = {
            # 控制词增强：增加常见 ASR 误识别词，提升触发灵敏度
            "forward": [
                "向前走", "往前走", "向前进", "往前进",
                "前进", "往前", "向前", "前移",
                "钱进", "前劲", "前景", "前镜", "前近", "前径"
            ],
            "backward": [
                "向后退", "往后退", "向后走", "往后走",
                "后退", "往后", "向后", "后移", "后撤", "退",
                "后腿", "后推"
            ],
            "left": ["左转", "向左", "往左", "左拐", "左边转", "左传"],
            "right": ["右转", "向右", "往右", "右拐", "右边转", "又转"],
            # 停止作为安全指令，补充更多同义词与英文词
            "stop": ["停止", "停下", "停住", "站住", "别动", "刹车", "别跑了", "不要跑了", "stop", "halt", "停"]
        }

        # 数量词映射
        self.number_map = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10
        }
        # 按关键词长度降序，避免短词抢占长词（如“停”抢“停止”）
        self._keyword_to_action = []
        for action, keywords in self.action_map.items():
            for kw in keywords:
                self._keyword_to_action.append((kw, action))
        self._keyword_to_action.sort(key=lambda x: len(x[0]), reverse=True)

    def _parse_number(self, text):
        num_match = re.search(r'(\d+|[一二两三四五六七八九十])\s*(步|米|个)?', text)
        if not num_match:
            return None
        num_str = num_match.group(1)
        if num_str.isdigit():
            return float(num_str)
        return float(self.number_map.get(num_str, 1.0))

    def _scan_actions_in_order(self, text):
        """
        从左到右扫描动作关键词，返回 [(action, start_idx, end_idx), ...]
        """
        results = []
        i = 0
        while i < len(text):
            matched = None
            for kw, action in self._keyword_to_action:
                if text.startswith(kw, i):
                    matched = (action, i, i + len(kw))
                    break
            if matched:
                # 去重：例如“往后退2米”会命中“往后”和“退”，二者属于同一动作短语
                if results:
                    prev_action, _prev_start, prev_end = results[-1]
                    if prev_action == matched[0] and (matched[1] - prev_end) <= 1:
                        i = matched[2]
                        continue
                results.append(matched)
                i = matched[2]
            else:
                i += 1
        return results

    def _scan_actions_fuzzy(self, text):
        """
        兜底模糊匹配：处理“左么转”“前径”等轻度 ASR 形变。
        """
        patterns = [
            (r'左.{0,2}转', "left"),
            (r'右.{0,2}转', "right"),
            (r'前.{0,2}(进|径|景|镜|近|劲)', "forward"),
            (r'后.{0,2}(退|推|腿)', "backward"),
            (r'(停止|停下|停住|站住|别动|刹车|别跑了|不要跑了|stop|halt|停)', "stop"),
        ]
        hits = []
        for pat, action in patterns:
            for m in re.finditer(pat, text):
                hits.append((action, m.start(), m.end()))
        hits.sort(key=lambda x: x[1])

        dedup = []
        last_end = -1
        for h in hits:
            if h[1] < last_end:
                continue
            dedup.append(h)
            last_end = h[2]
        return dedup

    def parse_sequence(self, text):
        """
        解析文本，返回动作序列:
        [
            {"action": "left", "distance": 1.0, "continuous": False},
            {"action": "forward", "distance": 3.0, "continuous": False},
        ]
        """
        text = (text or "").strip().lower()
        if not text:
            return []

        action_hits = self._scan_actions_in_order(text)
        if not action_hits:
            action_hits = self._scan_actions_fuzzy(text)
        if not action_hits:
            return []

        commands = []
        for idx, (action, _start, end) in enumerate(action_hits):
            next_start = action_hits[idx + 1][1] if idx + 1 < len(action_hits) else len(text)
            window = text[end:next_start]
            num = self._parse_number(window)

            continuous = False
            distance = 1.0
            if action in ("forward", "backward"):
                if num is not None:
                    distance = num
                else:
                    # 仅说“前进/后退”时，进入连续模式，直到“停止”
                    if re.search(r'(一直|持续|不停|连续)', text) or len(action_hits) == 1:
                        continuous = True
            elif action == "stop":
                distance = 0.0
            else:
                # 转向类动作默认一次执行
                distance = 1.0

            commands.append({
                "action": action,
                "distance": float(distance),
                "continuous": continuous
            })

        return commands

    def parse(self, text):
        """
        兼容旧接口，返回首个动作 (action, distance)
        """
        commands = self.parse_sequence(text)
        if not commands:
            return None, None
        first = commands[0]
        return first["action"], first["distance"]

if __name__ == "__main__":
    # 简单测试
    parser = CommandParser()
    tests = ["往前走三步", "向左转", "往后退2米", "停止", "右转前进三步", "前进"]
    for t in tests:
        print(f"输入: {t} -> 解析结果: {parser.parse_sequence(t)}")
