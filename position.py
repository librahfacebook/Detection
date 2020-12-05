# -*- encoding: utf-8 -*-
# @Time: 2020/12/05 10:00
# @Author: librah
# @Description: 
# @File: position.py
# @Version: 1.0

BORDER = [[142, 171], [101, 339], [283, 339], [296, 171]]

# 求取向量叉乘
def get_vector_cross_product(position0, position1, position):

    product_value = (position1[0]-position0[0]) * (position[1]-position0[1]) -       (position1[1]-position0[1])*(position[0]-position0[0])

    return product_value

# 判断该点是否在四边形内部
def isPosition(center_position):

    directions = []
    isPosition = True
    for i in range(0, len(BORDER)):
        direction = get_vector_cross_product(BORDER[i], BORDER[(i+1)%len(BORDER)], center_position)
        directions.append(direction)

    for i in range(0, len(directions)-1):
        if directions[i]*directions[i+1] < 0:
            isPosition = False
            break
    
    return isPosition


if __name__ == '__main__':
    print(isPosition((143,186)))
    
    
    
    

    
