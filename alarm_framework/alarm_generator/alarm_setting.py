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
        self.on_delay = on_delay
        self.off_delay = off_delay

    def __repr__(self):
        return "Limit: %f\nType: %s\nProcess Variable: %s\n\n" \
               % (self.limit, self.alm_type, self.proc_var)
