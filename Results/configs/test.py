import config

import dynamics as conf_dynamic

print("static stuff")
print(config.T)
print(config.variants)

print("dynamic stuff")
print(conf_dynamic.T)
print(conf_dynamic.IS_DEBUG)