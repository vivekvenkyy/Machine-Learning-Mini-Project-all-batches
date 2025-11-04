import { Point, Metrics } from '../types';

// Helper to compute euclidean distance between two points
const euclideanDistance = (p1: Point, p2: Point): number => {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
};

// Helper to compute centroid of a cluster
const computeClusterCentroid = (points: Point[]): Point => {
  const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
};

// Computes cluster-wise metrics (within-cluster sum of squares, between-cluster sum of squares)
const computeClusterStats = (data: Point[]): { clusters: Map<number, Point[]>; centroids: Map<number, Point>; globalCentroid: Point } => {
  const clusters = new Map<number, Point[]>();
  
  // Group points by cluster
  data.forEach(p => {
    const c = p.cluster ?? -1;
    if (!clusters.has(c)) clusters.set(c, []);
    clusters.get(c)!.push(p);
  });
  
  // Compute centroids for each cluster
  const centroids = new Map<number, Point>();
  clusters.forEach((points, c) => {
    if (c !== -1) { // Skip noise points
      centroids.set(c, computeClusterCentroid(points));
    }
  });
  
  // Compute global centroid (excluding noise points)
  const nonNoisePoints = data.filter(p => (p.cluster ?? -1) !== -1);
  const globalCentroid = computeClusterCentroid(nonNoisePoints);
  
  return { clusters, centroids, globalCentroid };
};

export const computeMetrics = (data: Point[]): Metrics => {
  const { clusters, centroids, globalCentroid } = computeClusterStats(data);
  
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
  
  // Calinski-Harabasz Index (Variance Ratio Criterion)
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
    
    // Average distance of points in cluster i to their centroid (cluster scatter)
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