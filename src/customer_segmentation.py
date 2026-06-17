# patch: disable threadpoolctl BEFORE sklearn imports
import threadpoolctl
from contextlib import contextmanager
threadpoolctl.threadpool_info = lambda: []
@contextmanager
def _noop_limits(limits=None, user_api=None):
    yield
threadpoolctl.threadpool_limits = _noop_limits

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

CLUSTER_FEATURES = [
    'lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'total_of_special_requests',
    'required_car_parking_spaces', 'is_repeated_guest',
]

def run_segmentation(n_clusters=4, sample_size=50000):
    os.makedirs('static/images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print(f'\u52a0\u8f7d\u6570\u636e...')
    df = pd.read_csv('data/processed/cleaned_data.csv')

    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    features = [c for c in CLUSTER_FEATURES if c in df_sample.columns]
    print(f'\u805a\u7c7b\u7279\u5f81 ({len(features)}): {features[:5]}...')

    X = df_sample[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f'\u8bad\u7ec3 MiniBatchKMeans n_clusters={n_clusters}...')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=3, max_iter=100)
    labels = kmeans.fit_predict(X_scaled)

    df_result = df_sample.copy()
    df_result['cluster'] = labels

    cluster_profiles = []
    for i in range(n_clusters):
        mask = labels == i
        cluster_df = df_result.iloc[mask]
        size = int(mask.sum())
        cancel_rate = float(cluster_df['is_canceled'].mean())
        profile = {'cluster': int(i), 'size': size, 'cancel_rate': round(cancel_rate, 4)}
        for feat in features[:8]:
            profile[feat] = round(float(cluster_df[feat].mean()), 2)
        cluster_profiles.append(profile)

    centroids = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids)
    centroid_df = pd.DataFrame(centroids_original, columns=features)
    centroid_df['cluster'] = range(n_clusters)

    labels_from_names = ['\u9ad8\u4ef7\u503c\u5ba2\u6237', '\u7ecf\u6d4e\u578b\u5ba2\u6237', '\u5546\u65c5\u5ba2\u6237', '\u5bb6\u5ead\u5ea6\u5047\u5ba2\u6237']
    if n_clusters == 4:
        sorted_idx = np.argsort(centroid_df['adr'].values)[::-1]
        cluster_names = [''] * n_clusters
        for rank, idx in enumerate(sorted_idx):
            cluster_names[idx] = labels_from_names[rank]
        for i, name in enumerate(cluster_names):
            df_result.loc[df_result['cluster'] == i, 'cluster_name'] = name
    else:
        df_result['cluster_name'] = df_result['cluster'].apply(lambda x: f'\u7fa4\u7ec4{x+1}')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:n_clusters]

    for idx in range(min(4, n_clusters)):
        ax = axes[idx // 2][idx % 2]
        feat_x = features[idx * 3 % len(features)]
        feat_y = features[(idx * 3 + 1) % len(features)]
        for ci in range(n_clusters):
            mask = labels == ci
            ax.scatter(X_scaled[mask, features.index(feat_x)],
                       X_scaled[mask, features.index(feat_y)],
                       c=colors[ci], alpha=0.3, s=5, label=f'\u7fa4\u7ec4{ci+1}')
        ax.scatter(centroids[:, features.index(feat_x)],
                   centroids[:, features.index(feat_y)],
                   c='black', marker='x', s=200, linewidths=3)
        ax.set_xlabel(feat_x, fontsize=11)
        ax.set_ylabel(feat_y, fontsize=11)
        ax.set_title(f'{feat_x} vs {feat_y}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('static/images/cluster_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\u805a\u7c7b\u6563\u70b9\u56fe\u5df2\u4fdd\u5b58')

    cluster_sizes = [p['size'] for p in cluster_profiles]
    cluster_rates = [p['cancel_rate'] * 100 for p in cluster_profiles]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cluster_profiles))
    width = 0.35
    bars1 = ax.bar(x - width/2, cluster_sizes, width, label='\u5ba2\u6237\u6570', color='#1f77b4', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, cluster_rates, width, label='\u53d6\u6d88\u7387%', color='#d62728', alpha=0.8)
    ax.set_xlabel('\u7fa4\u7ec4', fontsize=12)
    ax.set_ylabel('\u5ba2\u6237\u6570', fontsize=12)
    ax2.set_ylabel('\u53d6\u6d88\u7387 (%)', fontsize=12)
    ax.set_title('\u5ba2\u6237\u7fa4\u7ec4\u5206\u5e03\u4e0e\u53d6\u6d88\u7387', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'\u7fa4\u7ec4{p["cluster"]+1}' for p in cluster_profiles], fontsize=11)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_sizes)*0.01,
                f'{int(bar.get_height()):,}', ha='center', fontsize=10)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig('static/images/cluster_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\u805a\u7c7b\u5206\u5e03\u56fe\u5df2\u4fdd\u5b58')

    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/kmeans_scaler.pkl')
    joblib.dump({'features': features, 'n_clusters': n_clusters}, 'models/kmeans_config.pkl')
    print(f'\u805a\u7c7b\u6a21\u578b\u5df2\u4fdd\u5b58')

    print(f'\n=== \u5ba2\u6237\u7fa4\u7ec4\u753b\u50cf ===')
    for p in cluster_profiles:
        print(f'\n\u7fa4\u7ec4 {p["cluster"]+1}: \u5ba2\u6237\u6570={p["size"]:,}, \u53d6\u6d88\u7387={p["cancel_rate"]:.2%}')
        for k, v in p.items():
            if k not in ('cluster', 'size', 'cancel_rate'):
                print(f'  {k}: {v}')

    return {
        'profiles': cluster_profiles,
        'centroids': centroid_df.to_dict('records'),
        'features': features,
        'n_clusters': n_clusters,
    }

def load_segmentation():
    kmeans = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/kmeans_scaler.pkl')
    config = joblib.load('models/kmeans_config.pkl')
    return kmeans, scaler, config

def predict_cluster(features: dict):
    kmeans, scaler, config = load_segmentation()
    df = pd.DataFrame([features])
    df = df[[c for c in config['features'] if c in df.columns]].fillna(0)
    X = scaler.transform(df.values)
    cluster = int(kmeans.predict(X)[0])
    return {'cluster': cluster, 'cluster_name': f'\u7fa4\u7ec4{cluster+1}'}

if __name__ == '__main__':
    run_segmentation(n_clusters=4)
