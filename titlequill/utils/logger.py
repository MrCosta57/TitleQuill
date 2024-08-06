
class Logger:
    
    def __init__(self): pass
    
    # TODO - Use something more sophisticated than print
    def info (self, mess: str): print(f" INFO: {mess}")
    def warn (self, mess: str): print(f" WARN: {mess}")
    def error(self, mess: str): print(f"ERROR: {mess}")
    
class SilentLogger(Logger):
    
    def __init__(self): super().__init__()
    
    def info (self, mess: str): pass
    def warn (self, mess: str): pass
    def error(self, mess: str): pass