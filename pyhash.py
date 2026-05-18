class fnv1_32:
    def __call__(self, value) -> int:
        if not isinstance(value, (bytes, bytearray)):
            value = str(value).encode("utf-8")
        h = 2166136261
        for byte in value:
            h = (h * 16777619) & 0xFFFFFFFF
            h ^= byte
        return h
