__all__: list[str] = ['sec_to_hms']


def sec_to_hms(seconds: float) -> str:
    from    datetime    import  timedelta
    return str(timedelta(seconds=seconds))