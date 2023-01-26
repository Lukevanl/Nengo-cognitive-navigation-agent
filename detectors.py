def red_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 0:
        return True
    return False
        
def green_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 1:
        return True
    return False
        
def blue_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 2:
        return True
    return False
        
def yellow_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 3:
        return True
    return False
        
        
def magenta_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 4:
        return True
    return False
        
def white_detector(x):
    red = x[:3]
    green = x[3:6]
    blue = x[6:9]
    yellow = x[9:12]
    magenta = x[12:15]
    white = x[15:18]
    color = x[18:21]
    dot_products = [x[i * 3:(i + 1) * 3] @ color for i in range((len(x) - 1) // 3 )]
    max_val = max(dot_products)
    index = dot_products.index(max_val)
    if index == 4:
        return True
    return False