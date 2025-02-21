import pandas as pd

def read_csv_file(file_path):
    """
    讀取指定路徑的 CSV 文件，並返回 DataFrame 物件。
    若讀取過程中發生錯誤，則印出錯誤訊息。
    """
    try:
        data = pd.read_csv(file_path)
        print("成功讀取資料!")
        return data
    except Exception as e:
        print("讀取CSV文件時出現錯誤:", e)
        return None

if __name__ == '__main__':
    # 請將這裡的路徑替換為你自己的CSV檔案路徑
    file_path = "covid.csv"
    df = read_csv_file(file_path)
    df = pd.concat([df.iloc[:, 0:1], df.iloc[:, 42:]], axis=1)
    #if df is not None:
    #    # 顯示前5筆資料以確認讀取結果
    #    print("資料的前5筆數據:")
    #    print(df.head())
    corr_matrix = df.corr()
    target_col = df.columns[-1]
    target_corr = corr_matrix[target_col]
    target_corr = target_corr.drop(target_col)
    top10_columns = target_corr.abs().sort_values(ascending=False).head(10)

    print("與目標欄位 '{}' 最相關的10個欄位:".format(target_col))
    print(top10_columns)