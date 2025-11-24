import wandb
import pandas as pd
from typing import List, Dict, Any

# 자신의 W&B 아이디
ENTITY = "bjy6614-postech" 

PROJECT = "online_rl"
# ex) ddpg-Humanoid-v5, 
GROUP_NAME = "ddpg-Walker2d-v5"
# ------------------

def get_runs_data(entity: str, project: str, group: str) -> List[pd.DataFrame]:

    print(f"[{group}] 그룹의 런들을 W&B에서 검색 중...")
    
    # 1. W&B API 초기화
    api = wandb.Api()
    
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    
    if not runs:
        print(f"경고: 그룹 '{group}'에 해당하는 런을 찾을 수 없습니다. 설정(ENTITY, PROJECT, GROUP_NAME)을 확인하세요.")
        return []

    print(f"총 {len(runs)}개의 런 데이터를 다운로드합니다.")
    
    all_histories = []
    
    for run in runs:
        history = run.history(keys=["global/steps", "eval/avg_return"])
        
        if not history.empty and "eval/avg_return" in history.columns:
            history['run_name'] = run.name
            history['group'] = run.group
            all_histories.append(history)
            
    return all_histories

def aggregate_stats(all_histories: List[pd.DataFrame]) -> pd.DataFrame:
    if not all_histories:
        return pd.DataFrame()
    
    return_dfs = [df[["global/steps", "eval/avg_return"]] for df in all_histories]
    
    for i, df in enumerate(return_dfs):
        df = df.rename(columns={"eval/avg_return": f"return_{i}"}, inplace=True)
    
    merged_df = return_dfs[0]
    for i in range(1, len(return_dfs)):
        merged_df = pd.merge(merged_df, return_dfs[i], on="global/steps", how="outer")

    merged_df = merged_df.sort_values(by="global/steps").ffill()

    return_columns = [col for col in merged_df.columns if col.startswith("return_")]
    
    stats_df = pd.DataFrame()
    stats_df["global/steps"] = merged_df["global/steps"]

    stats_df["mean_return"] = merged_df[return_columns].mean(axis=1)
    stats_df["std_return"] = merged_df[return_columns].std(axis=1).fillna(0) # 런이 1개일 경우 std는 0
    
    return stats_df


def plot_stats(stats_df: pd.DataFrame, group_name: str):
    if stats_df.empty:
        print("분석할 통계 데이터가 없습니다.")
        return

    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # 1. 평균 리턴 곡선 플롯
        plt.plot(stats_df["global/steps"], stats_df["mean_return"], label=f"Mean Return ({group_name})", color="blue")
        
        # 2. 표준편차 영역 (신뢰 구간) 플롯
        # 평균 ± 표준편차로 신뢰 영역을 나타냅니다.
        lower_bound = stats_df["mean_return"] - stats_df["std_return"]
        upper_bound = stats_df["mean_return"] + stats_df["std_return"]
        plt.fill_between(stats_df["global/steps"], lower_bound, upper_bound, 
                         color="blue", alpha=0.15, label="Mean $\pm$ Std Dev")

        # 3. 그래프 레이블 및 제목 설정
        plt.title(f"Evaluation Performance: {group_name} (Across Multiple Seeds)", fontsize=16)
        plt.xlabel("Total Steps (Global/steps)", fontsize=14)
        plt.ylabel("Average Return (Mean $\pm$ Std)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # 4. 결과 저장
        output_filename = f"{group_name}_eval_plot.png"
        plt.savefig(output_filename)
        print(f"\n✅ 분석 결과가 '{output_filename}' 파일로 저장되었습니다.")
        plt.show()

    except ImportError:
        print("\n경고: Matplotlib이 설치되어 있지 않아 그래프를 그릴 수 없습니다.")
        print("데이터프레임 계산은 완료되었습니다. (Pandas 필요)")
        print("\n--- 통계 데이터 미리보기 ---")
        print(stats_df.head())


if __name__ == "__main__":
    if wandb is None:
        print("오류: wandb 라이브러리가 설치되어 있어야 실행할 수 있습니다.")
    else:
        run_histories = get_runs_data(ENTITY, PROJECT, GROUP_NAME)
        
        if run_histories:
            stats_result = aggregate_stats(run_histories)
            
            plot_stats(stats_result, GROUP_NAME)