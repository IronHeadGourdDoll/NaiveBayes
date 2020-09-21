
i = 0
count1, count2 = 0, 0
with open('2017081267_预测结果.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        _, re = line.split(' ')
        if i < 1000:
            if (re == "好评\n"):
                count1 += 1
        else:
            if (re == "差评\n"):
                count2 += 1
        i += 1
        line = f.readline()

print(count1/1000)
print(count2/1000)


