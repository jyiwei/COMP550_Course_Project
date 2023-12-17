import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
            self.__dict__.update(config)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)