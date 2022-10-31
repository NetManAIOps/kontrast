class TimeSpan:
    def __init__(self, start: int, end: int) -> None:
        """
        A span of time.
        Args:
            start:      Start 10-digit timestamp.
            end:        End 10-digit timestamp.
        """
        self.start: int = int(start)
        self.end: int = int(end)
