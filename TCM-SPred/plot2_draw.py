# --------------------------------------------------
#  1. Imports
# --------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pypinyin import pinyin, Style

# --------------------------------------------------
#  2. Helper: fallback pinyin
# --------------------------------------------------
def convert_to_pinyin(text):
    return ' '.join([''.join(item[0]) for item in pinyin(text, style=Style.NORMAL)])

# --------------------------------------------------
#  3. Hard-coded Latin mapping (spaces preserved)
# --------------------------------------------------
LATIN_MAP = {
    '木香': r'$\it{AR*}$',
    '地黄': r'$\it{PR}$',
    '白芍': r'$\it{PRA}$',
    '白及': r'$\it{BR}$',
    '陈皮': r'$\it{CRP}$',
    '知母': r'$\it{AR}$',
    '牡蛎': r'$\it{OC}$',
    '大血藤': r'$\it{SC}$',
    '山茱萸': r'$\it{CF}$',
    '浙贝': r'$\it{FTB}$'
}

# --------------------------------------------------
#  4. Main plotting function
# --------------------------------------------------
def plot_top10_herbs_rankings(total_scores_path, output_folder):
    Renal_fibrosis_symptoms = [
        '浮肿', '乏力', '腹痛', '腹胀', '呼吸困难', '咳嗽', '纳差', '呕吐',
        '皮肤瘙痒', '四肢痛', '头痛', '头晕', '胃脘嘈杂', '消瘦', '心慌',
        '胸痛', '多汗', '多尿', '烦躁', '关节不利', '关节痛', '汗出',
        '活动不利', '气促', '情绪不稳', '身痛', '下肢无力', '血尿',
        '厌食', '易怒'
    ]
    total_symptoms = len(Renal_fibrosis_symptoms)

    top10_df = pd.read_csv(total_scores_path).head(10)
    top10_herbs = top10_df['herb'].tolist()

    # Latin labels
    top10_labels = [LATIN_MAP.get(h, convert_to_pinyin(h)) for h in top10_herbs]

    rankings = {h: {'rank1': 0, 'rank2-5': 0, 'rank6-15': 0, 'rank>15': 0}
                for h in top10_herbs}

    for symptom in Renal_fibrosis_symptoms:
        file_path = os.path.join("predictions/Renal_fibrosis_prediction", f"Renal_fibrosis_{symptom}_scores.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            continue
        df = pd.read_csv(file_path).sort_values('score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        for h in top10_herbs:
            r = df.loc[df['herb'] == h, 'rank'].values
            if len(r) > 0:
                rank = r[0]
                if rank == 1:
                    rankings[h]['rank1'] += 1
                elif 2 <= rank <= 5:
                    rankings[h]['rank2-5'] += 1
                elif 6 <= rank <= 15:
                    rankings[h]['rank6-15'] += 1
                else:
                    rankings[h]['rank>15'] += 1
            else:
                rankings[h]['rank>15'] += 1

    rank1     = [rankings[h]['rank1']    for h in top10_herbs]
    rank2_5   = [rankings[h]['rank2-5']  for h in top10_herbs]
    rank6_15  = [rankings[h]['rank6-15'] for h in top10_herbs]
    rank_gt15 = [rankings[h]['rank>15']  for h in top10_herbs]

    colors = ['#e36146', '#2bae9e', '#65a2d2', '#bc90be']

    plt.figure(figsize=(16, 10))
    x = np.arange(len(top10_labels))
    width = 0.8

    p1 = plt.bar(x, rank1, width, color=colors[0], edgecolor='white', linewidth=1.2)
    p2 = plt.bar(x, rank2_5, width, bottom=rank1, color=colors[1], edgecolor='white', linewidth=1.2)
    p3 = plt.bar(x, rank6_15, width,
                 bottom=np.array(rank1) + np.array(rank2_5),
                 color=colors[2], edgecolor='white', linewidth=1.2)
    p4 = plt.bar(x, rank_gt15, width,
                 bottom=np.array(rank1) + np.array(rank2_5) + np.array(rank6_15),
                 color=colors[3], edgecolor='white', linewidth=1.2)

    plt.title('Rank Distribution of Top 10 Herbs in Renal fibrosis Symptoms',
              fontsize=28, fontweight='bold', pad=20)
    plt.ylabel('Number of Symptoms', fontsize=24)
    plt.xticks(x, top10_labels, rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)

    # Unified white labels
    def add_labels(rects):
        for rect in rects:
            h = rect.get_height()
            ypos = rect.get_y() + h / 2
            if h > 0:
                plt.text(rect.get_x() + rect.get_width() / 2., ypos, str(h),
                         ha='center', va='center', fontsize=19,
                         color='black', fontweight='bold')
    add_labels(p1); add_labels(p2); add_labels(p3); add_labels(p4)

    plt.legend(['Rank 1', 'Rank 2-5', 'Rank 6-15', 'Rank >15'],
               title='Rank Range', loc='upper right', fontsize=14, title_fontsize=15)

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "Top10_Herbs_Rank_Distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.show()
    return rankings

# --------------------------------------------------
#  5. Entry point
# --------------------------------------------------
if __name__ == "__main__":
    total_scores_path = "predictions/Renal_fibrosis_prediction/Renal_fibrosis_total_ranking.csv"
    output_folder = "predictions/Renal_fibrosis_prediction/figures"
    rankings = plot_top10_herbs_rankings(total_scores_path, output_folder)

    print("\nTop 10 herbs rank distribution across 30 Renal fibrosis symptoms:")
    for herb, stats in rankings.items():
        label = LATIN_MAP.get(herb, convert_to_pinyin(herb))
        print(f"{herb} ({label}): "
              f"Rank1: {stats['rank1']}, Ranks2-5: {stats['rank2-5']}, "
              f"Ranks6-15: {stats['rank6-15']}, Ranks>15: {stats['rank>15']}")