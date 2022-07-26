import pandas as pd
from icecream import ic
from PIL import Image

# 읽어올 엑셀 파일 지정
# filename = 'C:/MyProject/kobert/kobert/data/한국어_단발성_대화_데이터셋.xlsx'
# emotions = ['공포', '놀람', '중립', '혐오']
# # 엑셀 파일 읽어 오기
# df = pd.read_excel(filename, engine='openpyxl')
# for i in emotions:
#     data = df[df['Emotion'].str.contains(f'{i}')].index
#     df.drop(data, inplace=True)
# ic(df)
# df.to_excel('C:/MyProject/kobert/kobert/data/한국어_감정_데이터셋.xlsx', index=False)


sad_books = ['C:/MyProject/kobert/imgs/기분을 관리하면 인생이 관리된다.jpg']
for i in sad_books:
    image = Image.open(f"{i}")
    
image.show()