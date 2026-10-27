import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = np.random.rand(150, 2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

model_data = {
    "cluster_centers": kmeans.cluster_centers_.tolist(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist()
}

with open("cluster_model.json", "w", encoding="utf-8") as f:
    json.dump(model_data, f, indent=4)

print("Modell erfolgreich trainiert und 'cluster_model.json' wurde erstellt!")
