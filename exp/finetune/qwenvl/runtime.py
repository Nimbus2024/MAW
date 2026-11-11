def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def resolve_attn_implementation(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "flash_attention_2" if flash_attn_available() else "sdpa"
