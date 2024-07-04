import json
import logging
import re
import sys
from typing import Dict, Tuple, List, Literal, Callable, Optional

import numpy as np
from numba import jit

from .utils import del_all

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jit
def _find_index(table: np.ndarray, val: np.uint16) -> int:
    for i in range(table.size):
        if table[i] == val:
            return i
    return -1

@jit
def _fast_replace(table: np.ndarray, text: bytes) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    result = np.frombuffer(text, dtype=np.uint16).copy()
    replaced_words = []
    for i in range(result.size):
        ch = result[i]
        p = _find_index(table[0], ch)
        if p >= 0:
            repl_char = table[1][p]
            result[i] = repl_char
            replaced_words.append((chr(ch), chr(repl_char)))
    return result, replaced_words

class Normalizer:
    def __init__(self, map_file_path: str):
        self.logger = logger
        self.normalizers: Dict[str, Callable[[str], str]] = {}
        self.homophones_map = self._load_homophones_map(map_file_path)
        self.coding = "utf-16-le" if sys.byteorder == "little" else "utf-16-be"
        self.accept_pattern = re.compile(r"[^\u4e00-\u9fffA-Za-z，。、,\. ]")
        self.sub_pattern = re.compile(r"\[uv_break\]|\[laugh\]|\[lbreak\]")
        self.chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")
        self.english_word_pattern = re.compile(r"\b[A-Za-z]+\b")
        self.character_simplifier = str.maketrans({
            "：": "，", "；": "，", "！": "。", "（": "，", "）": "，", 
            "【": "，", "】": "，", "『": "，", "』": "，", "「": "，", 
            "」": "，", "《": "，", "》": "，", "－": "，", "‘": "", 
            "“": "", "’": "", "”": "", ":": ",", ";": ",", "!": ".", 
            "(": ",", ")": ",", "[": ",", "]": ",", ">": ",", "<": ",", 
            "-": ","
        })
        self.halfwidth_2_fullwidth = str.maketrans({
            "!": "！", '"': "“", "'": "‘", "#": "＃", "$": "＄", 
            "%": "％", "&": "＆", "(": "（", ")": "）", ",": "，", 
            "-": "－", "*": "＊", "+": "＋", ".": "。", "/": "／", 
            ":": "：", ";": "；", "<": "＜", "=": "＝", ">": "＞", 
            "?": "？", "@": "＠", "\\": "＼", "^": "＾", "`": "｀", 
            "{": "｛", "|": "｜", "}": "｝", "~": "～"
        })

    def __call__(self, text: str, do_text_normalization: bool = True, do_homophone_replacement: bool = True, lang: Optional[Literal["zh", "en"]] = None) -> str:
        if do_text_normalization:
            _lang = self._detect_language(text) if lang is None else lang
            if _lang in self.normalizers:
                text = self.normalizers[_lang](text)
            if _lang == "zh":
                text = self._apply_half2full_map(text)
        invalid_characters = self._count_invalid_characters(text)
        if invalid_characters:
            self.logger.warning(f"Found invalid characters: {invalid_characters}")
            text = self._apply_character_map(text)
        if do_homophone_replacement:
            arr, replaced_words = _fast_replace(self.homophones_map, text.encode(self.coding))
            if replaced_words:
                text = arr.tobytes().decode(self.coding)
                repl_res = ", ".join([f"{old}->{new}" for old, new in replaced_words])
                self.logger.info(f"Replaced homophones: {repl_res}")
        return text

    def register(self, name: str, normalizer: Callable[[str], str]) -> bool:
        if name in self.normalizers:
            self.logger.warning(f"Name {name} has already been registered.")
            return False
        try:
            val = normalizer("test string 测试字符串")
            if not isinstance(val, str):
                self.logger.warning("Normalizer must have the signature (str) -> str.")
                return False
        except Exception as e:
            self.logger.warning(f"Exception during registration: {e}")
            return False
        self.normalizers[name] = normalizer
        return True

    def unregister(self, name: str):
        if name in self.normalizers:
            del self.normalizers[name]

    def destroy(self):
        del_all(self.normalizers)
        del self.homophones_map

    def _load_homophones_map(self, map_file_path: str) -> np.ndarray:
        with open(map_file_path, "r", encoding="utf-8") as f:
            homophones_map: Dict[str, str] = json.load(f)
        map_arr = np.empty((2, len(homophones_map)), dtype=np.uint32)
        for i, (key, value) in enumerate(homophones_map.items()):
            map_arr[:, i] = (ord(key), ord(value))
        return map_arr

    def _count_invalid_characters(self, text: str) -> set:
        text = self.sub_pattern.sub("", text)
        return set(self.accept_pattern.findall(text))

    def _apply_half2full_map(self, text: str) -> str:
        return text.translate(self.halfwidth_2_fullwidth)

    def _apply_character_map(self, text: str) -> str:
        return text.translate(self.character_simplifier)

    def _detect_language(self, sentence: str) -> Literal["zh", "en"]:
        chinese_chars = len(self.chinese_char_pattern.findall(sentence))
        english_words = len(self.english_word_pattern.findall(sentence))
        return "zh" if chinese_chars > english_words else "en"
