import configparser
from functools import partial

config = configparser.ConfigParser()
config.read("config.ini")

get = partial(config.get, "DEFAULT")
get_bool = config["DEFAULT"].getboolean