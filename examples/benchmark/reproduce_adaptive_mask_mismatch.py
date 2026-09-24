"""Reproduce a cached-mask/native-acceptance inconsistency found during profiling."""
import json
from pathlib import Path
import xgrammar as xgr
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[2]
tokenizer = AutoTokenizer.from_pretrained(root / "profiling-results/full/tokenizer", local_files_only=True)
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer), max_threads=1, cache_enabled=False)
matcher = xgr.GrammarMatcher(compiler.compile_builtin_json_grammar())
prefix = '{"items":[1,{"text":"hello'
for token in tokenizer.encode(prefix, add_special_tokens=False):
    assert matcher.accept_token(token)
mask = xgr.allocate_token_bitmask(1, len(tokenizer))
matcher.fill_next_token_bitmask(mask)
token = 42785
allowed = bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
accepted = matcher.accept_token(token)
suffix = 'next":0}'
suffix_accepted = matcher.accept_string(suffix)
print(json.dumps({"prefix": prefix, "token_id": token, "token_text": tokenizer.decode([token]),
                  "mask_allows": allowed, "native_accepts": accepted,
                  "suffix_accepted": suffix_accepted, "completed": matcher.is_completed(),
                  "json_value": json.loads(prefix + tokenizer.decode([token]) + suffix)}, indent=2))
