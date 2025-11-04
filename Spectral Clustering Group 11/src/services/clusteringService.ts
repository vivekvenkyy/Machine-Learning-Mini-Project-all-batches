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

// Fixed seed for reproducible behavior. If you want variability, expose a setter.
const rng = mulberry32(0xC0FFEE);

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


// --- MOCK Clustering Algorithms & Metrics ---
// In a real application, these would run on a server. Here we simulate the output.

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
    // Sort points by x coordinate and take evenly spaced points as initial centroids
    const sortedData = [...data].sort((a, b) => a.x - b.x);
    const stride = Math.floor(data.length / k);
    const means = Array.from({ length: k }, (_, i) => sortedData[i * stride]);
    
    // Assign each point to the closest centroid
    return data.map(point => {
        let closestMeanIndex = 0;
        let minDistance = Infinity;
        means.forEach((mean, index) => {
            const distance = Math.sqrt(Math.pow(point.x - mean.x, 2) + Math.pow(point.y - mean.y, 2));
            if (distance < minDistance) {
                minDistance = distance;
                closestMeanIndex = index;
            }
        });
        return { ...point, cluster: closestMeanIndex };
    });
};

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
  // Helper function to compute kNN distance
  const kNearestNeighborsDistance = (point: Point, points: Point[], k: number): number => {
    return points
      .filter(p => p !== point)
      .map(p => euclideanDistance(point, p))
      .sort((a, b) => a - b)
      .slice(0, k)
      .reduce((sum, dist) => sum + dist, 0) / k;
  };

  // Compute intra-cluster density
  const computeClusterDensity = (points: Point[]): number => {
    if (points.length < 2) return 0;
    const k = Math.min(5, points.length - 1);
    return points.reduce((sum, p) => 
      sum + 1 / (1 + kNearestNeighborsDistance(p, points, k)), 0
    ) / points.length;
  };

  // Modified distance function for non-convex shapes
  const adaptiveDistance = (p1: Point, p2: Point, cluster: Point[]): number => {
    const directDist = euclideanDistance(p1, p2);
    if (directDist < 0.1) return directDist;

    const k = Math.min(7, Math.max(3, Math.floor(Math.sqrt(cluster.length))));
    const p1Neighbors = new Set(
      cluster
        .filter(p => p !== p1)
        .sort((a, b) => euclideanDistance(p1, a) - euclideanDistance(p1, b))
        .slice(0, k)
    );
    const p2Neighbors = new Set(
      cluster
        .filter(p => p !== p2)
        .sort((a, b) => euclideanDistance(p2, a) - euclideanDistance(p2, b))
        .slice(0, k)
    );

    const sharedNeighbors = [...p1Neighbors].filter(p => p2Neighbors.has(p));
    if (sharedNeighbors.length > 0) {
      return directDist * (1 - 0.5 * (sharedNeighbors.length / k));
    }
    return directDist;
  };

  const clusters = new Map<number, Point[]>();
  data.forEach(p => {
    const c = p.cluster ?? -1;
    if (!clusters.has(c)) clusters.set(c, []);
    clusters.get(c)!.push(p);
  });

  const centroids = new Map<number, Point>();
  const clusterDensities = new Map<number, number>();
  
  clusters.forEach((points, c) => {
    if (c !== -1) {
      centroids.set(c, computeClusterCentroid(points));
      clusterDensities.set(c, computeClusterDensity(points));
    }
  });

  const nonNoisePoints = data.filter(p => (p.cluster ?? -1) !== -1);
  if (nonNoisePoints.length === 0) {
    return { silhouette: 0, calinskiHarabasz: 0, daviesBouldin: 0 };
  }
  const globalCentroid = computeClusterCentroid(nonNoisePoints);

  // Silhouette coefficient with adaptive distance
  const silhouetteScores = data.map(point => {
    const c = point.cluster ?? -1;
    if (c === -1) return 0;

    const pointsInCluster = clusters.get(c)!;
    if (pointsInCluster.length <= 1) return 0;

    // a(i): Average adaptive distance to points in same cluster
    const a = pointsInCluster
      .filter(p => p !== point)
      .reduce((sum, other) => sum + adaptiveDistance(point, other, pointsInCluster), 0)
      / (pointsInCluster.length - 1);

    // b(i): Minimum average adaptive distance to other clusters
    let b = Infinity;
    clusters.forEach((otherPoints, otherC) => {
      if (otherC !== c && otherC !== -1) {
        const density = clusterDensities.get(otherC)!;
        const avgDist = otherPoints.reduce((sum, other) => 
          sum + adaptiveDistance(point, other, otherPoints), 0
        ) / otherPoints.length;
        // Weight distance by cluster density
        const weightedDist = avgDist * (1 + 0.5 * (1 - density));
        b = Math.min(b, weightedDist);
      }
    });

    if (!isFinite(b) || (a === 0 && b === 0)) return 0;
    return (b - a) / Math.max(a, b);
  });

  const silhouette = silhouetteScores.reduce((sum, s) => sum + s, 0) / silhouetteScores.length;

  // Calinski-Harabasz Index with density weighting
  let bss = 0;
  let wss = 0;
  let n = 0;

  centroids.forEach((centroid, c) => {
    const clusterPoints = clusters.get(c)!;
    const density = clusterDensities.get(c)!;
    n += clusterPoints.length;

    // Weight between-cluster sum by density
    const dx = centroid.x - globalCentroid.x;
    const dy = centroid.y - globalCentroid.y;
    bss += clusterPoints.length * (dx * dx + dy * dy) * (1 + density);

    // Use adaptive distance for within-cluster sum
    clusterPoints.forEach(point => {
      wss += Math.pow(adaptiveDistance(point, centroid, clusterPoints), 2);
    });
  });

  const k = centroids.size;
  const calinskiHarabasz = k <= 1 ? 0 : (bss * (n - k)) / (wss * (k - 1));

  // Davies-Bouldin Index with adaptive distances
  let dbSum = 0;
  const clusterIds = Array.from(centroids.keys());

  clusterIds.forEach(i => {
    const clusteri = clusters.get(i)!;
    const centroidi = centroids.get(i)!;

    // Use adaptive distance for cluster dispersion
    const si = clusteri.reduce((sum, p) => 
      sum + adaptiveDistance(p, centroidi, clusteri), 0
    ) / clusteri.length;

    let maxRij = 0;
    clusterIds.forEach(j => {
      if (i !== j) {
        const clusterj = clusters.get(j)!;
        const centroidj = centroids.get(j)!;
        const sj = clusterj.reduce((sum, p) => 
          sum + adaptiveDistance(p, centroidj, clusterj), 0
        ) / clusterj.length;

        // Weight inter-cluster distance by average density
        const avgDensity = (clusterDensities.get(i)! + clusterDensities.get(j)!) / 2;
        const mij = euclideanDistance(centroidi, centroidj) * (1 - 0.3 * avgDensity);
        if (mij > 0) {
          const rij = (si + sj) / mij;
          maxRij = Math.max(maxRij, rij);
        }
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
      const radius_threshold = 0.75; // Between inner (0.5) and outer (1.0) radius
      clusteredData = data.map(p => {
          const dist = Math.sqrt(p.x * p.x + p.y * p.y);
          return { ...p, cluster: dist > radius_threshold ? 0 : 1 };
      });
      break;
    default: // Blobs or Custom
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
    default: // Blobs or Custom
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
    default: // Blobs or Custom
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
  // Default min_samples is 5 or ln(n) for larger datasets
  const min_samples = Math.max(5, Math.ceil(Math.log(data.length)));
  
  // Helper function to get neighbors within eps distance
  const getNeighbors = (point: Point, points: Point[]): Point[] => {
    return points.filter(p => 
      p !== point && 
      Math.sqrt(Math.pow(p.x - point.x, 2) + Math.pow(p.y - point.y, 2)) <= params.eps
    );
  };

  // Initialize all points as unvisited (-2 means unvisited)
  let clusteredData: Point[] = data.map(p => ({ ...p, cluster: -2 }));
  let currentCluster = 0;

  // Process each point
  for (let i = 0; i < clusteredData.length; i++) {
    const point = clusteredData[i];
    
    // Skip if point has been processed
    if (point.cluster !== -2) continue;

    // Find neighbors
    const neighbors = getNeighbors(point, clusteredData);

    // If point doesn't have enough neighbors, mark as noise (-1)
    if (neighbors.length < min_samples) {
      point.cluster = -1;
      continue;
    }

    // Start a new cluster
    point.cluster = currentCluster;
    
    // Process neighbors
    let neighborQueue = [...neighbors];
    while (neighborQueue.length > 0) {
      const currentPoint = neighborQueue.pop()!;
      
      // If point was noise, add it to cluster
      if (currentPoint.cluster === -1) {
        currentPoint.cluster = currentCluster;
      }
      
      // If point hasn't been processed
      if (currentPoint.cluster === -2) {
        currentPoint.cluster = currentCluster;
        
        // Get neighbors of current point
        const currentNeighbors = getNeighbors(currentPoint, clusteredData);
        
        // If current point is a core point, add its unprocessed neighbors to queue
        if (currentNeighbors.length >= min_samples) {
          neighborQueue.push(...currentNeighbors.filter(n => 
            n.cluster === -2 || n.cluster === -1
          ));
        }
      }
    }
    
    currentCluster++;
  }

  return {
    algorithm: 'DBSCAN',
    data: clusteredData,
    metrics: computeMetrics(clusteredData),
    params: { ...params, min_samples }, // Include min_samples in params
  };
};

// --- CSV Parsing ---
export const parseCSV = (file: File, options: { dropLastColumn: boolean }): Promise<Point[]> => {
  return new Promise((resolve, reject) => {
    // Check file size first
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      reject(new Error("File is too large. Please use a CSV file smaller than 10MB."));
      return;
    }

    const reader = new FileReader();
    
    // Set timeout for reading
    const timeout = setTimeout(() => {
      reader.abort();
      reject(new Error("File reading timed out. Please try a smaller file."));
    }, 30000); // 30 second timeout

    reader.onload = async (event) => {
      clearTimeout(timeout);
      try {
        const text = event.target?.result as string;
        if (!text || text.trim() === '') {
          throw new Error("The CSV file is empty.");
        }

        // Split by any common line ending
        const rows = text.split(/\r\n|\n|\r/).filter(row => row.trim() !== '');
        
        if (rows.length < 2) {
          throw new Error("CSV file must contain a header and at least one data row.");
        }

        // Validate header format
        const header = rows[0].split(',').map(h => h.trim());
        if (header.length < 2) {
          throw new Error("CSV must have at least two columns.");
        }

        // Process data rows with progress tracking
        const batchSize = 1000;
        const processedRows: number[][] = [];
        
        for (let i = 1; i < rows.length; i += batchSize) {
          const batch = rows.slice(i, i + batchSize);
          const processedBatch = batch.map(row => {
            const values = row.split(',').map(v => {
              const parsed = parseFloat(v.trim());
              if (isNaN(parsed)) {
                throw new Error(`Invalid numeric value found in row ${i}: ${v}`);
              }
              return parsed;
            });
            
            return options.dropLastColumn ? values.slice(0, -1) : values;
          });
          
          processedRows.push(...processedBatch);
          
          // Allow UI to update every batch
          if (i + batchSize < rows.length) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        if (processedRows.length === 0) {
          throw new Error("No valid numeric data rows found in the CSV file.");
        }

        const numFeatures = processedRows[0].length;
        if (numFeatures < 2) {
          throw new Error(`The processed data has only ${numFeatures} feature(s). At least two are required for 2D visualization.`);
        }

        if (numFeatures > 2) {
          console.warn(`Dataset has ${numFeatures} features. Using first two features for visualization.`);
        }

        // Normalize the data to prevent extreme values
        const points: Point[] = processedRows.map(row => {
          const x = Math.max(-100, Math.min(100, row[0])); // Clamp between -100 and 100
          const y = Math.max(-100, Math.min(100, row[1]));
          return { x, y };
        });

        resolve(points);
      } catch (error: any) {
        reject(new Error(`Error parsing CSV: ${error.message}`));
      }
    };

    reader.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("Failed to read the file. Please make sure it's a valid CSV file."));
    };

    reader.onabort = () => {
      clearTimeout(timeout);
      reject(new Error("File reading was aborted."));
    };

    reader.readAsText(file);
  });
};

// --- Spectral helpers (affinity matrix, spectral embedding, silhouette scores) ---

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
  // For each point, compute a(i) = avg intra-cluster distance, b(i) = min avg distance to other clusters
  const n = clusteredData.length;
  if (n === 0) return [];
  const clusters = new Map<number, Point[]>();
  clusteredData.forEach((p) => {
    const c = p.cluster ?? -1;
    if (!clusters.has(c)) clusters.set(c, []);
    clusters.get(c)!.push(p);
  });

  const d = (p1: Point, p2: Point) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

  const scores: number[] = clusteredData.map((p, idx) => {
    const c = p.cluster ?? -1;
    const same = clusters.get(c) || [];
    if (same.length <= 1) return 0; // silhouette undefined for singleton
    const a = same.reduce((s, q) => s + d(p, q), 0) / (same.length - 1);
    let b = Infinity;
    for (const [otherC, pts] of clusters.entries()) {
      if (otherC === c) continue;
      if (otherC === -1) continue; // skip noise cluster when calculating b
      const avg = pts.reduce((s, q) => s + d(p, q), 0) / pts.length;
      if (avg < b) b = avg;
    }
    if (!isFinite(b)) return 0;
    return (b - a) / Math.max(a, b);
  });

  return scores;
};