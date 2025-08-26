"""
生成关于自定义token的freeze训练数据，提升模型对于新增token的基础理解。
建立<ROW_X>和行、行坐标, X, X-224, abs(X-224)等token的关联
建立<COL_Y>和列、列坐标, Y, Y-224, abs(Y-224)等token的关联


1. <ROW_X>等token的数据，使用一下几种模板：
    A
        input: <ROW_X>是什么意思？|介绍一下<ROW_X>
        output: <ROW_X>是布局网格的第X行的行坐标，是布局网格的倒数第abs(X-224)行，非负索引为X，负数索引为X-224。布局网格是指芯片布局规划时用的矩形网格，将一张画布切分成224 * 224个网格。
    B
        input: 布局网格的第X行是什么？|布局网格的第X行的行坐标是什么？|布局网格的第X行的坐标是什么？
        output: 是<ROW_X> | 布局网格的第X行是<ROW_X> 
    C 
        input: 布局网格的倒数第abs(X-224)行是什么？| 布局网格的倒数第abs(X-224)行的行坐标是什么？|布局网格的倒数第abs(X-224)行的坐标是什么
        output: 是<ROW_X> | 布局网格的倒数第abs(X-224)行是<ROW_X> 
    D
        input: 布局网格[非负]索引为X的行是什么？|布局网格[非负]索引为X的行坐标是什么？
        output: 是<ROW_X> | 布局网格中非负索引为X的行坐标是<ROW_X> 
    E
        input: 布局网格负数索引为X-224的行是什么？|布局网格负数索引为X-224的行坐标是什么？
        output: 是<ROW_X> | 布局网格中负数索引为X-224的行坐标是<ROW_X> 

2. <COL_Y>等token的数据，使用类似<ROW_X>的模板，并进行相应的属性替换

3. (<ROW_X>, <COL_Y>)类型数据
    A
        input: (<ROW_X>, <COL_Y>)是什么意思？|介绍一下(<ROW_X>, <COL_Y>)
        output: (<ROW_X>, <COL_Y>)表示布局网格中第X行第Y列的网格，是第X行与第Y列的交点。| (<ROW_X>, <COL_Y>)表示布局网格中的一个网格，它的行坐标是<ROW_X>，列坐标是<COL_Y>。
    B
        input: 布局网格中行坐标为<ROW_X>，列坐标为<COL_Y>的网格的坐标是什么？
        output: 是(<ROW_X>, <COL_Y>)。|布局网格中行坐标为<ROW_X>，列坐标为<COL_Y>的网格的坐标是(<ROW_X>, <COL_Y>)。
    C
        input: 第X行第Y列的网格的坐标是什么？
        output: 是(<ROW_X>, <COL_Y>)。|第X行第Y列的网格的坐标是(<ROW_X>, <COL_Y>)。

4. <ORIENT_N>相关
"""

import json
import random

def pick_one(choices:list[str]) -> str:
    return random.choice(choices)

def optional(value: str, p=0.5) -> str:
    return value if random.random() < p else ''

def generate_row_data():
    """
    1. <ROW_X>等token的数据，使用一下几种模板：
    A
        input: <ROW_X>是什么意思？|介绍一下<ROW_X>
        output: <ROW_X>是布局网格的第X行的行坐标，是布局网格的倒数第abs(X-224)行，非负索引为X，负数索引为X-224。
    B
        input: 布局网格的第X行是什么？|布局网格的第X行的行坐标是什么？|布局网格的第X行的坐标是什么？
        output: 是<ROW_X> | 布局网格的第X行是<ROW_X> 
    C 
        input: 布局网格的倒数第abs(X-224)行是什么？| 布局网格的倒数第abs(X-224)行的行坐标是什么？|布局网格的倒数第abs(X-224)行的坐标是什么
        output: 是<ROW_X> | 布局网格的倒数第abs(X-224)行是<ROW_X> 
    D
        input: 布局网格[非负]索引为X的行是什么？|布局网格[非负]索引为X的行坐标是什么？
        output: 是<ROW_X> | 布局网格中非负索引为X的行坐标是<ROW_X> 
    E
        input: 布局网格负数索引为X-224的行是什么？|布局网格负数索引为X-224的行坐标是什么？
        output: 是<ROW_X> | 布局网格中负数索引为X-224的行坐标是<ROW_X> 
    """

    data = []
    for x in range(224):
        # 模板A
        data.append({
            "instruction": "",
            "input": pick_one([f"<ROW_{x}>是什么意思{optional('？')}", f"介绍一下<ROW_{x}>"]),
            "output": f"<ROW_{x}>是布局网格的第{x}行的行坐标，也是布局网格的倒数第{abs(x-224)}行，对应的非负索引为{x}，负数索引为{x-224}。{optional('布局网格是指芯片布局规划时用的矩形网格，将一张画布切分成224 * 224个网格。', p=0.01)}"
        })
        # 模板B
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格的第{x}行是什么{optional('？')}", f"布局网格的第{x}行的行坐标是什么{optional('？')}", f"布局网格的第{x}行的坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<ROW_{x}>。", f"布局网格的第{x}行是<ROW_{x}>。"])
        })
        # 模板C
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格的倒数第{abs(x-224)}行是什么{optional('？')}", f"布局网格的倒数第{abs(x-224)}行的行坐标是什么{optional('？')}", f"布局网格的倒数第{abs(x-224)}的坐标是什么"]),
            "output": pick_one([f"是<ROW_{x}>。", f"布局网格的倒数第{abs(x-224)}行是<ROW_{x}>。"])
        })
        # 模板D
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格{optional('非负')}索引为{x}的行是什么{optional('？')}", f"布局网格{optional('非负')}索引为{x}的行坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<ROW_{x}>。", f"布局网格中非负索引为{x}的行坐标是<ROW_{x}>。"])
        })
        # 模板E
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格负数索引为{x-224}的行是什么{optional('？')}", f"布局网格负数索引为{x-224}的行坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<ROW_{x}>。", f"布局网格中负数索引为{x-224}的行坐标是<ROW_{x}>。"])
        })
    return data


def generate_col_data():
    """
    1. <COL_Y>等token的数据，使用一下几种模板：
    A
        input: <COL_Y>是什么意思？|介绍一下<COL_Y>
        output: <COL_Y>是布局网格的第Y列的列坐标，是布局网格的倒数第abs(Y-224)列，非负索引为Y，负数索引为Y-224。
    B
        input: 布局网格的第Y列是什么？|布局网格的第Y列的列坐标是什么？|布局网格的第Y列的坐标是什么？
        output: 是<COL_Y> | 布局网格的第Y列是<COL_Y> 
    C 
        input: 布局网格的倒数第abs(Y-224)列是什么？| 布局网格的倒数第abs(Y-224)列的列坐标是什么？|布局网格的倒数第abs(Y-224)列的坐标是什么
        output: 是<COL_Y> | 布局网格的倒数第abs(Y-224)列是<COL_Y> 
    D
        input: 布局网格[非负]索引为Y的列是什么？|布局网格[非负]索引为Y的列坐标是什么？
        output: 是<COL_Y> | 布局网格中非负索引为Y的列坐标是<COL_Y> 
    E
        input: 布局网格负数索引为Y-224的列是什么？|布局网格负数索引为Y-224的列坐标是什么？
        output: 是<COL_Y> | 布局网格中负数索引为Y-224的列坐标是<COL_Y> 
    """

    data = []
    for y in range(224):
        # 模板A
        data.append({
            "instruction": "",
            "input": pick_one([f"<COL_{y}>是什么意思{optional('？')}", f"介绍一下<COL_{y}>"]),
            "output": f"<COL_{y}>是布局网格的第{y}列的列坐标，也是布局网格的倒数第{abs(y-224)}列，对应的非负索引为{y}，负数索引为{y-224}。{optional('布局网格是指芯片布局规划时用的矩形网格，将一张画布切分成224 * 224个网格。', p=0.01)}"
        })
        # 模板B
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格的第{y}列是什么{optional('？')}", f"布局网格的第{y}列的列坐标是什么{optional('？')}", f"布局网格的第{y}列的坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<COL_{y}>。", f"布局网格的第{y}列是<COL_{y}>。"])
        })
        # 模板C
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格的倒数第{abs(y-224)}列是什么{optional('？')}", f"布局网格的倒数第{abs(y-224)}列的列坐标是什么{optional('？')}", f"布局网格的倒数第{abs(y-224)}的坐标是什么"]),
            "output": pick_one([f"是<COL_{y}>。", f"布局网格的倒数第{abs(y-224)}列是<COL_{y}>。"])
        })
        # 模板D
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格{optional('非负')}索引为{y}的列是什么{optional('？')}", f"布局网格{optional('非负')}索引为{y}的列坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<COL_{y}>。", f"布局网格中非负索引为{y}的列坐标是<COL_{y}>。"])
        })
        # 模板E
        data.append({
            "instruction": "",
            "input": pick_one([f"布局网格负数索引为{y-224}的列是什么{optional('？')}", f"布局网格负数索引为{y-224}的列坐标是什么{optional('？')}"]),
            "output": pick_one([f"是<COL_{y}>。", f"布局网格中负数索引为{y-224}的列坐标是<COL_{y}>。"])
        })
    return data


def generate_rc_data():
    """
    (<ROW_X>, <COL_Y>)类型数据
    A
        input: (<ROW_X>, <COL_Y>)是什么意思？|介绍一下(<ROW_X>, <COL_Y>)
        output: (<ROW_X>, <COL_Y>)表示布局网格中第X行第Y列的网格，是第X行与第Y列的交点。| (<ROW_X>, <COL_Y>)表示布局网格中的一个网格，它的行坐标是<ROW_X>，列坐标是<COL_Y>。
    B
        input: 布局网格中行坐标为<ROW_X>，列坐标为<COL_Y>的网格的坐标是什么？
        output: 是(<ROW_X>, <COL_Y>)。|布局网格中行坐标为<ROW_X>，列坐标为<COL_Y>的网格的坐标是(<ROW_X>, <COL_Y>)。
    C
        input: 第X行第Y列的网格的坐标是什么？
        output: 是(<ROW_X>, <COL_Y>)。|第X行第Y列的网格的坐标是(<ROW_X>, <COL_Y>)。
    """
    data = []
    for x in range(224):
        for y in range(224):
            # 模板A
            data.append({
                "instruction": "",
                "input": pick_one([f"(<ROW_{x}>, <COL_{y}>)是什么意思{optional('？')}", f"介绍一下(<ROW_{x}>, <COL_{y}>)"]),
                "output": pick_one([f"(<ROW_{x}>, <COL_{y}>)表示布局网格中第{x}行第{y}列的网格，是第{x}行与第{y}列的交点。", f"(<ROW_{x}>, <COL_{y}>)表示布局网格中的一个网格，它的行坐标是<ROW_{x}>，列坐标是<COL_{y}>。"])
            })
            # 模板B
            data.append({
                "instruction": "",
                "input": f"布局网格中行坐标为<ROW_{x}>，列坐标为<COL_{y}>的网格的坐标是什么{optional('？')}",
                "output": pick_one([f"是(<ROW_{x}>, <COL_{y}>)。", f"布局网格中行坐标为<ROW_{x}>，列坐标为<COL_{y}>的网格的坐标是(<ROW_{x}>, <COL_{y}>)。"])
            })
            # 模板C
            data.append({
                "instruction": "",
                "input": f"第{x}行第{y}列的网格的坐标是什么{optional('？')}",
                "output": pick_one([f"是(<ROW_{x}>, <COL_{y}>)。", f"第{x}行第{y}列的网格的坐标是(<ROW_{x}>, <COL_{y}>)。"])
            })

    return random.sample(data, int(len(data)*0.01))



if __name__ == "__main__":
    row_data = generate_row_data()
    col_data = generate_col_data()
    rc_data = generate_rc_data()

    total_data = row_data + col_data + rc_data
    print(len(total_data))


    with open('./data/fpllm_token_identity.jsonl', 'w') as fp:
        for data in total_data:
            fp.write(json.dumps(data, ensure_ascii=False)+'\n')