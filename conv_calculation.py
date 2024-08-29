import math

#conversion rate for King County, Washington USA:
#conv= 111*cos(lat)
angle_deg = 47.6
angle_rad = math.radians(angle_deg)
conv_factor_kc = 111 * math.cos(angle_rad) #1 degree of latitude difference-74.85km for King County