import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("dark")
sns.color_palette("Set2")

def preprocess(row):
    row['Metric'] = row['Metric'].replace('ROC AUC','ROC-AUC')
    row['Metric'] = row['Metric'].replace('roc_auc','ROC-AUC')
    row['Metric'] = row['Metric'].replace('f1_score','F1 Score')
    row['Metric'] = row['Metric'].replace('recall','Recall')
    row['Metric'] = row['Metric'].replace('precision','Precision')
    row['Metric'] = row['Metric'].replace('accuracy','Accuracy')
    return row

summary = pd.DataFrame()
for round in range(0,10):
    for fold in range(0,4):
        for approach in ['Ours','QChecker']:
            file_middle_name='metrics' if approach == 'Ours' else 'Qchecker'
            filename = f'./{approach}/{round}{file_middle_name}{fold}.csv'
            for idx, row in pd.read_csv(filename).iterrows():
                row = preprocess(row)
                row['Approach'] = approach
                summary = summary._append(row, ignore_index=True)

plt.figure(figsize=(8, 3))
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=8)

sns.boxplot(data=summary, x='Metric', y='Value', hue='Approach', palette='Paired')

plt.xlabel('')

plt.ylabel('')
plt.legend(loc='center left', fontsize='small')

plt.grid()
plt.tight_layout()
plt.savefig('boxplot.pdf')
plt.show()