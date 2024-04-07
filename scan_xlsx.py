import pandas as pd

# Function to correctly process each row into the desired format
def process_row_correctly(row):
    row_data = []
    row_number = row.name + 3  # Adjust row index to match the xlsx file's row numbers starting from 3

    # Process side A (columns 0 to 5)
    for col in range(0, 6, 2):  # Skip every other column to match the pairs
        if pd.notna(row[col]) and pd.notna(row[col+1]):
            row_data.append([row_number, row[col], 'A', row[col+1]])

    # Process side B (columns 6 to 11)
    for col in range(6, 12, 2):  # Skip every other column to match the pairs
        if pd.notna(row[col]) and pd.notna(row[col+1]):
            row_data.append([row_number, row[col], 'B', row[col+1]])

    # Get the outcome from the 13th column (index 12)
    outcome = row[12] if pd.notna(row[12]) else None

    return (row_data, outcome)

# Load the xlsx file
file_path = r'F:\code\arkbvb\battles\愚人节统计表.xlsx'

# Read the xlsx file, skipping the first two header rows
df = pd.read_excel(file_path, header=None, skiprows=2)

# Apply the function to each row and collect the results in a list
formatted_data = [process_row_correctly(row) for index, row in df.iterrows()]

# (Optional) If you want to display the formatted data
# for i, row in enumerate(formatted_data, start=3):
#     print(f"Row {i}: {row}")


lookup_table = {
    '拳击宗师': 14,
    '衣架射手': 20,
    '小寄居蟹': 3,
    '冰手术师': 1,
    '苦难的具象': 17,
    '杰斯顿': 10,
    '保鲜膜骑士': 2,
    '庞贝': 0,
    '镜子机关枪': 6,
    '巧克力流心虫虫': 15,
    '砸人的石头': 13,
    '奔跑吧躯壳': 11,
    '迟钝的持盾者': 24,
    '扎人的石头': 19,
    '劈柴骑士': 12,
    '迫击炮弹投手': 16,
    '弧光武士': 7,
    '火苗与软钢': 4,
    '流鼻涕虫虫': 23,
    '源石的腿脚': 5,
    '责罚者': 22,
    '锁链拳手': 8,
    '普通的萨卡兹': 21,
    '扩音术师': 25,
    '窃笑鳄鱼': 9,
    '狗Pro': 18
}

# 替换reformatted_data中的中文字符串为数字
reformatted_data_with_numbers = []
for data_row, outcome in formatted_data:
    # Replace Chinese characters with numbers using the lookup table
    new_data_row = [[row[0], lookup_table.get(row[1], row[1]), row[2], row[3]] for row in data_row]
    reformatted_data_with_numbers.append((new_data_row, outcome))

# 显示前几行数据以验证输出
for item in reformatted_data_with_numbers:
    print(str(item))  # 为了简洁只显示前三行数据