import { DataPoint } from './ClusteringDashboard';

// K-Means clustering algorithm
export function kMeans(
  data: DataPoint[],
  k: number,
  maxIterations: number = 100
): { data: DataPoint[]; centers: { x: number; y: number }[] } {
  const points = data.map(d => ({ x: d.x, y: d.y }));
  
  // Initialize centroids randomly
  const centroids = [...points]
    .sort(() => Math.random() - 0.5)
    .slice(0, k);

  let assignments = new Array(points.length).fill(0);
  
  for (let iter = 0; iter < maxIterations; iter++) {
    // Assign points to nearest centroid
    const newAssignments = points.map(point => {
      let minDist = Infinity;
      let closestCentroid = 0;
      
      centroids.forEach((centroid, idx) => {
        const dist = Math.sqrt(
          (point.x - centroid.x) ** 2 + (point.y - centroid.y) ** 2
        );
        if (dist < minDist) {
          minDist = dist;
          closestCentroid = idx;
        }
      });
      
      return closestCentroid;
    });

    // Check for convergence
    if (JSON.stringify(assignments) === JSON.stringify(newAssignments)) {
      break;
    }
    
    assignments = newAssignments;

    // Update centroids
    for (let i = 0; i < k; i++) {
      const clusterPoints = points.filter((_, idx) => assignments[idx] === i);
      if (clusterPoints.length > 0) {
        centroids[i] = {
          x: clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length,
          y: clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length,
        };
      }
    }
  }

  const clusteredData = data.map((point, idx) => ({
    ...point,
    cluster: assignments[idx],
  }));

  return { data: clusteredData, centers: centroids };
}

// DBSCAN clustering algorithm
export function dbscan(
  data: DataPoint[],
  eps: number,
  minSamples: number
): { data: DataPoint[]; numClusters: number } {
  const points = data.map(d => ({ x: d.x, y: d.y }));
  const labels = new Array(points.length).fill(-1);
  let clusterId = 0;

  const getNeighbors = (pointIdx: number): number[] => {
    const neighbors: number[] = [];
    const point = points[pointIdx];
    
    points.forEach((otherPoint, idx) => {
      if (idx === pointIdx) return;
      const dist = Math.sqrt(
        (point.x - otherPoint.x) ** 2 + (point.y - otherPoint.y) ** 2
      );
      if (dist <= eps) {
        neighbors.push(idx);
      }
    });
    
    return neighbors;
  };

  const expandCluster = (pointIdx: number, neighbors: number[], clusterId: number) => {
    labels[pointIdx] = clusterId;
    
    for (let i = 0; i < neighbors.length; i++) {
      const neighborIdx = neighbors[i];
      
      if (labels[neighborIdx] === -1) {
        labels[neighborIdx] = clusterId;
        const neighborNeighbors = getNeighbors(neighborIdx);
        
        if (neighborNeighbors.length >= minSamples) {
          neighbors.push(...neighborNeighbors);
        }
      }
    }
  };

  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue;
    
    const neighbors = getNeighbors(i);
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Noise point
    } else {
      expandCluster(i, neighbors, clusterId);
      clusterId++;
    }
  }

  const clusteredData = data.map((point, idx) => ({
    ...point,
    cluster: labels[idx],
  }));

  return { data: clusteredData, numClusters: clusterId };
}

// Simplified BIRCH clustering algorithm
export function birch(
  data: DataPoint[],
  k: number
): { data: DataPoint[]; subclusterCenters: { x: number; y: number }[] } {
  const points = data.map(d => ({ x: d.x, y: d.y }));
  
  // Phase 1: Build CF tree (simplified - just create subclusters)
  const numSubclusters = Math.min(k * 3, points.length);
  const subclusterSize = Math.max(1, Math.floor(points.length / numSubclusters));
  
  const subclusters: { x: number; y: number }[] = [];
  
  for (let i = 0; i < numSubclusters; i++) {
    const start = i * subclusterSize;
    const end = Math.min(start + subclusterSize, points.length);
    const subclusterPoints = points.slice(start, end);
    
    if (subclusterPoints.length > 0) {
      const center = {
        x: subclusterPoints.reduce((sum, p) => sum + p.x, 0) / subclusterPoints.length,
        y: subclusterPoints.reduce((sum, p) => sum + p.y, 0) / subclusterPoints.length,
      };
      subclusters.push(center);
    }
  }

  // Phase 2: Cluster the subclusters using K-Means
  const subclusterData = subclusters.map((sc, idx) => ({
    x: sc.x,
    y: sc.y,
    cluster: 0,
  }));
  
  const { data: clusteredSubclusters } = kMeans(subclusterData, k);
  
  // Phase 3: Assign original points to clusters
  const assignments = points.map(point => {
    let minDist = Infinity;
    let closestSubcluster = 0;
    
    subclusters.forEach((subcluster, idx) => {
      const dist = Math.sqrt(
        (point.x - subcluster.x) ** 2 + (point.y - subcluster.y) ** 2
      );
      if (dist < minDist) {
        minDist = dist;
        closestSubcluster = idx;
      }
    });
    
    return clusteredSubclusters[closestSubcluster].cluster;
  });

  const clusteredData = data.map((point, idx) => ({
    ...point,
    cluster: assignments[idx],
  }));

  return {
    data: clusteredData,
    subclusterCenters: subclusters,
  };
}
