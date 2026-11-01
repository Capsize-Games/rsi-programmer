def synth_4af19495e56c6b1b(x: int, y: int) -> int:
    n0 = x + y
    n1 = x + n0
    n2 = x * n1
    n3 = n2 - y
    return n3
