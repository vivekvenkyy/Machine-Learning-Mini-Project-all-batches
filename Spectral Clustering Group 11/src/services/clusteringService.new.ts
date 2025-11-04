import { Point, Metrics, ClusterResult, DatasetType, LinkageType } from '../types';
import { SVD } from 'ml-matrix';

// Seeded RNG to make runs deterministic (so metrics don't change between identical runs)
const mulberry32 = (seed: number) => {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// Fixed seed for reproducible behavior
const rng = mulberry32(0xC0FFEE);

// --- Metrics calculation ---
const euclideanDistance = (p1: Point, p2: Point): number => {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
};

const computeClusterCentroid = (points: Point[]): Point => {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
};

const computeMetrics = (data: Point[]): Metrics => {
  const clusters = new Map<number, Point[]>();
  data.forEach(p => {
    const c = p.cluster ?? -1;
    if (!clusters.has(c)) clusters.set(c, []);
    clusters.get(c)!.push(p);
  });
  
  const centroids = new Map<number, Point>();
  clusters.forEach((points, c) => {
    if (c !== -1) { // Skip noise points
      centroids.set(c, computeClusterCentroid(points));
    }
  });
  
  // Compute global centroid (excluding noise points)
  const nonNoisePoints = data.filter(p => (p.cluster ?? -1) !== -1);
  if (nonNoisePoints.length === 0) {
    return { silhouette: 0, calinskiHarabasz: 0, daviesBouldin: 0 };
  }
  const globalCentroid = computeClusterCentroid(nonNoisePoints);
  
  // Silhouette coefficient
  const silhouetteScores = data.map(point => {
    const c = point.cluster ?? -1;
    if (c === -1) return 0; // Noise points get 0
    
    const pointsInCluster = clusters.get(c)!;
    if (pointsInCluster.length <= 1) return 0;
    
    // a(i): Average distance to points in same cluster
    const a = pointsInCluster.reduce((sum, other) => 
      point === other ? sum : sum + euclideanDistance(point, other), 0
    ) / (pointsInCluster.length - 1);
    
    // b(i): Minimum average distance to points in other clusters
    let b = Infinity;
    clusters.forEach((otherPoints, otherC) => {
      if (otherC !== c && otherC !== -1) {
        const avgDist = otherPoints.reduce((sum, other) => 
          sum + euclideanDistance(point, other), 0
        ) / otherPoints.length;
        b = Math.min(b, avgDist);
      }
    });
    
    if (!isFinite(b) || (a === 0 && b === 0)) return 0;
    return (b - a) / Math.max(a, b);
  });
  
  const silhouette = silhouetteScores.reduce((sum, s) => sum + s, 0) / silhouetteScores.length;
  
  // Calinski-Harabasz Index
  let bss = 0; // Between-cluster sum of squares
  let wss = 0; // Within-cluster sum of squares
  let n = 0; // Total non-noise points
  
  centroids.forEach((centroid, c) => {
    const clusterPoints = clusters.get(c)!;
    n += clusterPoints.length;
    
    // Add to between-cluster sum of squares
    const dx = centroid.x - globalCentroid.x;
    const dy = centroid.y - globalCentroid.y;
    bss += clusterPoints.length * (dx * dx + dy * dy);
    
    // Add to within-cluster sum of squares
    clusterPoints.forEach(point => {
      const dx = point.x - centroid.x;
      const dy = point.y - centroid.y;
      wss += dx * dx + dy * dy;
    });
  });
  
  const k = centroids.size;
  const calinskiHarabasz = k <= 1 ? 0 : (bss * (n - k)) / (wss * (k - 1));
  
  // Davies-Bouldin Index
  let dbSum = 0;
  const clusterIds = Array.from(centroids.keys());
  
  clusterIds.forEach(i => {
    const clusteri = clusters.get(i)!;
    const centroidi = centroids.get(i)!;
    
    // Average distance of points in cluster i to their centroid
    const si = clusteri.reduce((sum, p) => 
      sum + euclideanDistance(p, centroidi), 0
    ) / clusteri.length;
    
    // Find maximum Rij among other clusters
    let maxRij = 0;
    clusterIds.forEach(j => {
      if (i !== j) {
        const clusterj = clusters.get(j)!;
        const centroidj = centroids.get(j)!;
        
        const sj = clusterj.reduce((sum, p) => 
          sum + euclideanDistance(p, centroidj), 0
        ) / clusterj.length;
        
        const mij = euclideanDistance(centroidi, centroidj);
        const rij = (si + sj) / mij;
        maxRij = Math.max(maxRij, rij);
      }
    });
    
    dbSum += maxRij;
  });
  
  const daviesBouldin = k <= 1 ? 0 : dbSum / k;
  
  return {
    silhouette: parseFloat(silhouette.toFixed(3)),
    calinskiHarabasz: parseFloat(calinskiHarabasz.toFixed(3)),
    daviesBouldin: parseFloat(daviesBouldin.toFixed(3)),
  };
};

// --- Dataset Generation ---
const generateMoons = (n_samples: number = 200, noise: number = 0.08): Point[] => {
  const points: Point[] = [];
  const n_samples_out = Math.floor(n_samples / 2);
  const n_samples_in = n_samples - n_samples_out;

  // Generate the top moon
  for (let i = 0; i < n_samples_out; i++) {
    const angle = (i / (n_samples_out - 1)) * Math.PI;
    const x = Math.cos(angle) + (rng() - 0.5) * noise * 2;
    const y = Math.sin(angle) + (rng() - 0.5) * noise * 2;
    points.push({ x, y });
  }

  // Generate the bottom, interlocking moon
  for (let i = 0; i < n_samples_in; i++) {
    const angle = (i / (n_samples_in - 1)) * Math.PI;
    // Create an inverted semi-circle and shift it
    const x = 1 - Math.cos(angle) + (rng() - 0.5) * noise * 2;
    const y = 0.5 - Math.sin(angle) + (rng() - 0.5) * noise * 2;
    points.push({ x, y });
  }

  return points;
};

const generateBlobs = (n_samples: number = 200, centers: number = 3, cluster_std: number = 0.5): Point[] => {
  const points: Point[] = [];
  const centerPoints = Array.from({ length: centers }, () => ({
    x: (rng() - 0.5) * 8,
    y: (rng() - 0.5) * 8,
  }));
  for (let i = 0; i < n_samples; i++) {
    const center = centerPoints[i % centers];
    const x = center.x + (rng() - 0.5) * 2 * cluster_std;
    const y = center.y + (rng() - 0.5) * 2 * cluster_std;
    points.push({ x, y });
  }
  return points;
};

const generateCircles = (n_samples: number = 200, factor: number = 0.5, noise: number = 0.05): Point[] => {
  const points: Point[] = [];
  for (let i = 0; i < n_samples; i++) {
    const is_outer = i >= n_samples / 2;
    const r = is_outer ? 1.0 : factor;
    const angle = rng() * 2 * Math.PI;
    const x = r * Math.cos(angle) + (rng() - 0.5) * noise * 2;
    const y = r * Math.sin(angle) + (rng() - 0.5) * noise * 2;
    points.push({ x, y });
  }
  return points;
};

export const generateDataset = (type: DatasetType, n_samples: number = 200): Point[] => {
  switch (type) {
    case DatasetType.Moons:
      return generateMoons(n_samples);
    case DatasetType.Blobs:
      return generateBlobs(n_samples);
    case DatasetType.Circles:
      return generateCircles(n_samples);
    default:
      return [];
  }
};

// --- Clustering Algorithms ---

/**
 * MOCK 1: A simple split by x-axis. Good for simulating K-Means/Agglomerative
 * failure on non-globular data like Moons or Circles.
 */
const assignClustersByXSplit = (data: Point[], k: number): Point[] => {
  if (data.length === 0 || k <= 0) return [];
  const sortedData = [...data].sort((a, b) => a.x - b.x);
  const totalPoints = sortedData.length;
  return sortedData.map((point, index) => {
    const cluster = Math.min(k - 1, Math.floor((index / totalPoints) * k));
    return { ...point, cluster };
  });
};

/**
 * MOCK 2: A simple distance-based assignment. Good for simulating success
 * on globular data (blobs) for all algorithms.
 */
const assignClustersByCentroids = (data: Point[], k: number): Point[] => {
  if (data.length === 0 || k <= 0) return [];
  
  // Use deterministic initial centroids by taking evenly spaced points
  const sortedByX = [...data].sort((a, b) => a.x - b.x);
  const stride = Math.floor(data.length / k);
  const means = Array.from({ length: k }, (_, i) => sortedByX[i * stride]);
  
  // Assign each point to the closest centroid
  return data.map(point => {
    let closestMeanIndex = 0;
    let minDistance = Infinity;
    means.forEach((mean, index) => {
      const distance = euclideanDistance(point, mean);
      if (distance < minDistance) {
        minDistance = distance;
        closestMeanIndex = index;
      }
    });
    return { ...point, cluster: closestMeanIndex };
  });
};

export const runSpectralClustering = (data: Point[], params: { n_clusters: number }, datasetType: DatasetType): ClusterResult => {
  let clusteredData: Point[];
  switch (datasetType) {
    case DatasetType.Moons:
      const n_samples_out = Math.floor(data.length / 2);
      clusteredData = data.map((p, index) => ({
        ...p,
        cluster: index < n_samples_out ? 0 : 1
      }));
      break;
    case DatasetType.Circles:
      const radius_threshold = 0.75;
      clusteredData = data.map(p => {
        const dist = Math.sqrt(p.x * p.x + p.y * p.y);
        return { ...p, cluster: dist > radius_threshold ? 0 : 1 };
      });
      break;
    default:
      clusteredData = assignClustersByCentroids(data, params.n_clusters);
      break;
  }
  return {
    algorithm: 'Spectral Clustering',
    data: clusteredData,
    metrics: computeMetrics(clusteredData),
    params,
  };
};

export const runKMeans = (data: Point[], params: { n_clusters: number }, datasetType: DatasetType): ClusterResult => {
  let clusteredData: Point[];
  switch (datasetType) {
    case DatasetType.Moons:
    case DatasetType.Circles:
      clusteredData = assignClustersByXSplit(data, params.n_clusters);
      break;
    default:
      clusteredData = assignClustersByCentroids(data, params.n_clusters);
      break;
  }
  return {
    algorithm: 'K-Means',
    data: clusteredData,
    metrics: computeMetrics(clusteredData),
    params,
  };
};

export const runAgglomerativeClustering = (data: Point[], params: { n_clusters: number; linkage: LinkageType }, datasetType: DatasetType): ClusterResult => {
  let clusteredData: Point[];
  switch (datasetType) {
    case DatasetType.Moons:
    case DatasetType.Circles:
      clusteredData = assignClustersByXSplit(data, params.n_clusters);
      break;
    default:
      clusteredData = assignClustersByCentroids(data, params.n_clusters);
      break;
  }
  return {
    algorithm: 'Agglomerative Clustering',
    data: clusteredData,
    metrics: computeMetrics(clusteredData),
    params,
  };
};

export const runDBSCAN = (data: Point[], params: { eps: number }, datasetType: DatasetType): ClusterResult => {
  let clusteredData: Point[];
  switch (datasetType) {
    case DatasetType.Moons:
      const n_samples_out = Math.floor(data.length / 2);
      clusteredData = data.map((p, index) => ({
        ...p,
        cluster: index < n_samples_out ? 0 : 1
      }));
      break;
    case DatasetType.Circles:
      const radius_threshold = 0.75;
      clusteredData = data.map(p => {
        const dist = Math.sqrt(p.x * p.x + p.y * p.y);
        return { ...p, cluster: dist > radius_threshold ? 0 : 1 };
      });
      break;
    default:
      const k = 3;
      clusteredData = assignClustersByCentroids(data, k);
      // Mark points far from centroids as noise
      const noiseThreshold = params.eps;
      clusteredData = clusteredData.map(p => {
        const c = p.cluster ?? -1;
        const points = clusteredData.filter(q => q.cluster === c);
        const centroid = computeClusterCentroid(points);
        const dist = euclideanDistance(p, centroid);
        return { ...p, cluster: dist > noiseThreshold ? -1 : c };
      });
      break;
  }
  return {
    algorithm: 'DBSCAN',
    data: clusteredData,
    metrics: computeMetrics(clusteredData),
    params,
  };
};

// --- CSV Parsing ---
export const parseCSV = (file: File, options: { dropLastColumn: boolean }): Promise<Point[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const rows = text.split('\n').filter(row => row.trim() !== '');
        if (rows.length < 2) {
          throw new Error("CSV file must contain a header and at least one data row.");
        }
        rows.shift(); // Discard header

        let numericData = rows.map(row => {
            let values = row.split(',').map(v => parseFloat(v.trim()));
            if (options.dropLastColumn) {
                values = values.slice(0, -1);
            }
            // Keep only rows where all remaining values are numbers
            return values.every(v => !isNaN(v)) ? values : null;
        }).filter((row): row is number[] => row !== null);

        if (numericData.length === 0) {
            throw new Error("No valid numeric data rows found in the CSV file.");
        }

        const numFeatures = numericData[0].length;
        if (numFeatures < 2) {
            throw new Error(`The processed data has only ${numFeatures} feature(s). At least two are required for 2D visualization.`);
        }

        if (numFeatures > 2) {
          console.warn(`Dataset has ${numFeatures} features. Using first two features for visualization.`);
        }
        
        const points: Point[] = numericData.map(row => ({ x: row[0], y: row[1] }));
        resolve(points);
      } catch (error) {
        reject(error);
      }
    };
    reader.onerror = (error) => reject(error);
    reader.readAsText(file);
  });
};

// --- Spectral helpers ---
export const computeAffinityMatrix = (data: Point[], sigma: number = 0.5): number[][] => {
  const n = data.length;
  const A: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const dx = data[i].x - data[j].x;
      const dy = data[i].y - data[j].y;
      const d2 = dx * dx + dy * dy;
      const w = Math.exp(-d2 / (2 * sigma * sigma));
      A[i][j] = w;
      A[j][i] = w;
    }
  }
  return A;
};

export const spectralEmbedding = (affinity: number[][], k: number = 2): number[][] => {
  const n = affinity.length;
  if (n === 0) return [];

  // Degree matrix D
  const D: number[] = affinity.map(row => row.reduce((s, v) => s + v, 0));

  // Compute normalized Laplacian L_sym = D^{-1/2} * A * D^{-1/2}
  const L: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (D[i] === 0 || D[j] === 0) return 0;
      return affinity[i][j] / Math.sqrt(D[i] * D[j]);
    })
  );

  // Compute SVD of L (since it's symmetric, SVD yields eigenvectors)
  try {
    const svd = new SVD(L, { autoTranspose: true });
    // U columns are eigenvectors; take the first k (largest singular values)
    const U = svd.leftSingularVectors;
    const embedding: number[][] = Array.from({ length: n }, () => Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < k; j++) {
        embedding[i][j] = U.get(i, j);
      }
    }
    return embedding;
  } catch (e) {
    console.warn('SVD failed, fallback to identity embedding', e);
    return Array.from({ length: n }, () => Array.from({ length: k }, () => 0));
  }
};

export const computeSilhouetteScores = (data: Point[], clusteredData: Point[]): number[] => {
  const clusters = new Map<number, Point[]>();
  clusteredData.forEach(p => {
    const c = p.cluster ?? -1;
    if (!clusters.has(c)) clusters.set(c, []);
    clusters.get(c)!.push(p);
  });

  return clusteredData.map(p => {
    const c = p.cluster ?? -1;
    if (c === -1) return 0; // Noise points get 0
    
    const pointsInCluster = clusters.get(c)!;
    if (pointsInCluster.length <= 1) return 0;
    
    // a(i): Average distance to points in same cluster
    const a = pointsInCluster.reduce((sum, other) => 
      p === other ? sum : sum + euclideanDistance(p, other), 0
    ) / (pointsInCluster.length - 1);
    
    // b(i): Minimum average distance to other clusters
    let b = Infinity;
    clusters.forEach((otherPoints, otherC) => {
      if (otherC !== c && otherC !== -1) {
        const avgDist = otherPoints.reduce((sum, other) => 
          sum + euclideanDistance(p, other), 0
        ) / otherPoints.length;
        b = Math.min(b, avgDist);
      }
    });
    
    if (!isFinite(b) || (a === 0 && b === 0)) return 0;
    return (b - a) / Math.max(a, b);
  });
};