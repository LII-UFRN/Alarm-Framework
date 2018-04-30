class AlarmSetting:
    limit = None
    alm_type = None
    proc_var = None
    on_delay = None
    off_delay = None

    def __init__(self, limit, alm_type, proc_var, on_delay=None, off_delay=None):
        self.limit = limit
        self.alm_type = alm_type
        self.proc_var = proc_var
        self.on_delay = 0 if on_delay is None else on_delay
        self.off_delay = 0 if off_delay is None else off_delay

    def as_dict(self):
        return {
            'limit': self.limit,
            'alm_type': self.alm_type,
            'proc_var': self.proc_var,
            'on_delay': self.on_delay,
            'off_delay': self.off_delay
        }
