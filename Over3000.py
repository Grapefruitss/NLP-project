import pandas as pd

# CSV 파일 경로 지정
file_path = 'crawled_data0.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path, header=None)  # header=None을 사용하여 첫 줄을 헤더로 간주하지 않음

# 길이가 2990자를 넘는 줄을 저장할 리스트
long_lines = []

# 각 줄 검사
for index, row in df.iterrows():
    line = row[0]  # 첫 번째 열의 값 (자소서 내용)
    if len(line) > 2990:
        long_lines.append((index, line))  # 인덱스와 함께 저장

# 결과를 저장할 리스트
split_lines = []

# 각 줄 처리
for idx, long_line in long_lines:
    # 2900번째 글자부터
    start_index = 2900
    substring = long_line[start_index:]
    
    # 첫 번째로 발견되는 '.'의 위치 찾기
    dot_index = substring.find('.')
    
    if dot_index != -1:
        # '.'의 위치를 기준으로 자소서 분할
        split_point = start_index + dot_index
        first_part = long_line[:split_point + 1]  # 처음부터 '.'까지
        second_part = long_line[split_point + 1:]  # '.' 다음부터 끝까지
    # else:
    #     # '.'이 발견되지 않으면 전체 자소서를 첫 번째 부분으로 처리
    #     first_part = long_line
    #     second_part = ""

    # 분할 자소서 리스트 추가
    split_lines.append([first_part])
    # if second_part:
    #     
    split_lines.append([second_part])

# new csv에 저장
output_file_path = 'split_crawled_data.csv'
split_df = pd.DataFrame(split_lines)
split_df.to_csv(output_file_path, index=False, header=False)

print(f"분할 자소 '{output_file_path}'에 저장")

# 
