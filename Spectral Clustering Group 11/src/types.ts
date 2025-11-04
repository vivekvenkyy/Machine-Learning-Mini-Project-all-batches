export interface Point {
  x: number;
  y: number;
  cluster?: number;
}

export interface Metrics {
  silhouette: number;
  calinskiHarabasz: number;
  daviesBouldin: number;
}

export interface ClusterResult {
  algorithm: string;
  data: Point[];
  metrics: Metrics;
  params: Record<string, any>;
}

export enum DatasetType {
  Moons = 'Moons',
  Blobs = 'Blobs',
  Circles = 'Circles',
  Custom = 'Custom',
}

export type LinkageType = 'ward' | 'complete' | 'average' | 'single';

export interface AlgorithmParameters {
  spectral: { n_clusters: number };
  kmeans: { n_clusters: number };
  agglomerative: { n_clusters: number; linkage: LinkageType };
  dbscan: { eps: number };
}