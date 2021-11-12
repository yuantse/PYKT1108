import random

# 計算BMI
def calculateBMI(height, weight):
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        return 'thin'
    elif bmi < 25:
        return 'normal'
    return 'fat'


# 產生BMI資料
# 開檔案準備寫入bmi.csv
with open('data/bmi.csv', 'w', encoding='UTF-8') as file1:
    # 定義資料欄位
    file1.write('height,weight,label\n')
    # 分類判斷初始值皆為0
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    # 產生50000筆資料
    for i in range(50000):
        # 隨機產生身高140~205
        currentHeight = random.randint(140, 205)
        # 隨機產生體重40~90
        currentWeight = random.randint(40, 90)
        # 計算BMI判斷分類
        label = calculateBMI(currentHeight, currentWeight)
        category[label] += 1
        file1.write("%d,%d,%s\n" % (currentHeight, currentWeight, label))

print("generate OK, result={}".format(category))