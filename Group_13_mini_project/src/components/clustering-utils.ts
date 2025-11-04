import { DataPoint } from './ClusteringDashboard';

// Box-Muller transform for normal distribution
function randomNormal(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * std + mean;
}

// Generate blob data similar to sklearn's make_blobs
export function generateBlobData(
  nSamples: number,
  nCenters: number,
  clusterStd: number,
  seed?: number
): DataPoint[] {
  if (seed !== undefined) {
    // Simple seeding (not cryptographically secure)
    Math.random = (() => {
      let s = seed;
      return () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
      };
    })();
  }

  const points: DataPoint[] = [];
  const samplesPerCenter = Math.floor(nSamples / nCenters);

  // Generate random centers
  const centers = Array.from({ length: nCenters }, () => ({
    x: Math.random() * 8 - 4,
    y: Math.random() * 8 - 4,
  }));

  // Generate points around each center
  for (let i = 0; i < nCenters; i++) {
    const center = centers[i];
    for (let j = 0; j < samplesPerCenter; j++) {
      points.push({
        x: randomNormal(center.x, clusterStd),
        y: randomNormal(center.y, clusterStd),
        cluster: i,
      });
    }
  }

  // Add remaining samples to random centers
  const remaining = nSamples - points.length;
  for (let i = 0; i < remaining; i++) {
    const center = centers[Math.floor(Math.random() * nCenters)];
    points.push({
      x: randomNormal(center.x, clusterStd),
      y: randomNormal(center.y, clusterStd),
      cluster: Math.floor(Math.random() * nCenters),
    });
  }

  return points;
}

// Generate concentric circles
export function generateCirclesData(nSamples: number): DataPoint[] {
  const points: DataPoint[] = [];
  const samplesPerCircle = Math.floor(nSamples / 2);
  
  // Inner circle
  for (let i = 0; i < samplesPerCircle; i++) {
    const angle = (i / samplesPerCircle) * 2 * Math.PI;
    const radius = 2 + randomNormal(0, 0.2);
    points.push({
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
      cluster: 0,
    });
  }
  
  // Outer circle
  for (let i = 0; i < nSamples - samplesPerCircle; i++) {
    const angle = (i / (nSamples - samplesPerCircle)) * 2 * Math.PI;
    const radius = 5 + randomNormal(0, 0.2);
    points.push({
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle),
      cluster: 1,
    });
  }
  
  return points;
}

// Generate two interleaving moons
export function generateMoonsData(nSamples: number): DataPoint[] {
  const points: DataPoint[] = [];
  const samplesPerMoon = Math.floor(nSamples / 2);
  
  // First moon
  for (let i = 0; i < samplesPerMoon; i++) {
    const angle = (i / samplesPerMoon) * Math.PI;
    const radius = 4;
    points.push({
      x: radius * Math.cos(angle) + randomNormal(0, 0.3),
      y: radius * Math.sin(angle) + randomNormal(0, 0.3),
      cluster: 0,
    });
  }
  
  // Second moon (inverted and shifted)
  for (let i = 0; i < nSamples - samplesPerMoon; i++) {
    const angle = (i / (nSamples - samplesPerMoon)) * Math.PI;
    const radius = 4;
    points.push({
      x: radius * Math.cos(angle + Math.PI) + randomNormal(0, 0.3) + 4,
      y: radius * Math.sin(angle + Math.PI) + randomNormal(0, 0.3) - 2,
      cluster: 1,
    });
  }
  
  return points;
}

// Generate spiral patterns
export function generateSpiralsData(nSamples: number): DataPoint[] {
  const points: DataPoint[] = [];
  const samplesPerSpiral = Math.floor(nSamples / 3);
  
  for (let s = 0; s < 3; s++) {
    const angleOffset = (s * 2 * Math.PI) / 3;
    for (let i = 0; i < samplesPerSpiral; i++) {
      const t = i / samplesPerSpiral;
      const angle = angleOffset + t * 4 * Math.PI;
      const radius = 0.5 + t * 4;
      points.push({
        x: radius * Math.cos(angle) + randomNormal(0, 0.2),
        y: radius * Math.sin(angle) + randomNormal(0, 0.2),
        cluster: s,
      });
    }
  }
  
  // Add remaining samples
  const remaining = nSamples - points.length;
  for (let i = 0; i < remaining; i++) {
    const s = Math.floor(Math.random() * 3);
    const angleOffset = (s * 2 * Math.PI) / 3;
    const t = Math.random();
    const angle = angleOffset + t * 4 * Math.PI;
    const radius = 0.5 + t * 4;
    points.push({
      x: radius * Math.cos(angle) + randomNormal(0, 0.2),
      y: radius * Math.sin(angle) + randomNormal(0, 0.2),
      cluster: s,
    });
  }
  
  return points;
}

// Generate grid pattern
export function generateGridData(nSamples: number): DataPoint[] {
  const points: DataPoint[] = [];
  const gridSize = 4;
  const spacing = 2;
  
  for (let i = 0; i < nSamples; i++) {
    const row = Math.floor(Math.random() * gridSize);
    const col = Math.floor(Math.random() * gridSize);
    points.push({
      x: (col - gridSize / 2) * spacing + randomNormal(0, 0.3),
      y: (row - gridSize / 2) * spacing + randomNormal(0, 0.3),
      cluster: row * gridSize + col,
    });
  }
  
  return points;
}

// Generate random uniform distribution
export function generateRandomData(nSamples: number): DataPoint[] {
  const points: DataPoint[] = [];
  
  for (let i = 0; i < nSamples; i++) {
    points.push({
      x: Math.random() * 10 - 5,
      y: Math.random() * 10 - 5,
      cluster: 0,
    });
  }
  
  return points;
}

// Generate dataset based on type
export function generateDataset(type: string, nSamples: number = 300): DataPoint[] {
  switch (type) {
    case 'blobs':
      return generateBlobData(nSamples, 3, 0.5);
    case 'circles':
      return generateCirclesData(nSamples);
    case 'moons':
      return generateMoonsData(nSamples);
    case 'spirals':
      return generateSpiralsData(nSamples);
    case 'grid':
      return generateGridData(nSamples);
    case 'random':
      return generateRandomData(nSamples);
    default:
      return generateBlobData(nSamples, 3, 0.5);
  }
}

// Add random outliers
export function addOutliers(count: number): DataPoint[] {
  const outliers: DataPoint[] = [];
  for (let i = 0; i < count; i++) {
    outliers.push({
      x: Math.random() * 12 - 6,
      y: Math.random() * 12 - 6,
      cluster: -1,
      isOutlier: true,
    });
  }
  return outliers;
}
