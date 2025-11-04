import { useEffect, useState } from 'react';
import { Card } from './ui/card';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { ScatterPlot } from './ScatterPlot';
import { kMeans, dbscan, birch } from './clustering-algorithms';
import { DataPoint } from './ClusteringDashboard';
import { Badge } from './ui/badge';

interface ClusteringPanelProps {
  algorithm: 'K-Means' | 'DBSCAN' | 'BIRCH';
  data: DataPoint[];
  setData: (data: DataPoint[]) => void;
  baseData: DataPoint[];
  numClusters?: number;
  onNumClustersChange?: (value: number) => void;
  eps?: number;
  minSamples?: number;
  onEpsChange?: (value: number) => void;
  onMinSamplesChange?: (value: number) => void;
  color: 'cyan' | 'purple' | 'pink';
  theme: 'light' | 'dark';
}

const colorMap = {
  cyan: {
    dark: {
      border: 'border-cyan-500/30',
      shadow: 'shadow-cyan-500/20',
      glow: 'shadow-[0_0_30px_rgba(34,211,238,0.3)]',
      text: 'text-cyan-400',
      bg: 'bg-cyan-500/10',
    },
    light: {
      border: 'border-cyan-300',
      shadow: 'shadow-cyan-200/40',
      glow: 'shadow-[0_0_20px_rgba(34,211,238,0.2)]',
      text: 'text-cyan-700',
      bg: 'bg-cyan-50',
    },
  },
  purple: {
    dark: {
      border: 'border-purple-500/30',
      shadow: 'shadow-purple-500/20',
      glow: 'shadow-[0_0_30px_rgba(168,85,247,0.3)]',
      text: 'text-purple-400',
      bg: 'bg-purple-500/10',
    },
    light: {
      border: 'border-purple-300',
      shadow: 'shadow-purple-200/40',
      glow: 'shadow-[0_0_20px_rgba(168,85,247,0.2)]',
      text: 'text-purple-700',
      bg: 'bg-purple-50',
    },
  },
  pink: {
    dark: {
      border: 'border-pink-500/30',
      shadow: 'shadow-pink-500/20',
      glow: 'shadow-[0_0_30px_rgba(236,72,153,0.3)]',
      text: 'text-pink-400',
      bg: 'bg-pink-500/10',
    },
    light: {
      border: 'border-pink-300',
      shadow: 'shadow-pink-200/40',
      glow: 'shadow-[0_0_20px_rgba(236,72,153,0.2)]',
      text: 'text-pink-700',
      bg: 'bg-pink-50',
    },
  },
};

export function ClusteringPanel({
  algorithm,
  data,
  setData,
  baseData,
  numClusters,
  onNumClustersChange,
  eps,
  minSamples,
  onEpsChange,
  onMinSamplesChange,
  color,
  theme,
}: ClusteringPanelProps) {
  const [centers, setCenters] = useState<{ x: number; y: number }[]>([]);
  const [numDetectedClusters, setNumDetectedClusters] = useState(0);
  const colors = colorMap[color][theme];

  useEffect(() => {
    if (baseData.length === 0) return;

    let clusteredData: DataPoint[];
    let detectedClusters = 0;
    let clusterCenters: { x: number; y: number }[] = [];

    switch (algorithm) {
      case 'K-Means':
        const kMeansResult = kMeans(baseData, numClusters || 3);
        clusteredData = kMeansResult.data;
        clusterCenters = kMeansResult.centers;
        detectedClusters = numClusters || 3;
        break;
      case 'DBSCAN':
        const dbscanResult = dbscan(baseData, eps || 0.6, minSamples || 5);
        clusteredData = dbscanResult.data;
        detectedClusters = dbscanResult.numClusters;
        break;
      case 'BIRCH':
        const birchResult = birch(baseData, numClusters || 3);
        clusteredData = birchResult.data;
        clusterCenters = birchResult.subclusterCenters;
        detectedClusters = numClusters || 3;
        break;
      default:
        clusteredData = baseData;
    }

    setData(clusteredData);
    setCenters(clusterCenters);
    setNumDetectedClusters(detectedClusters);
  }, [baseData, numClusters, eps, minSamples, algorithm]);

  return (
    <Card className={`${colors.border} ${colors.shadow} ${colors.glow} border-2 ${
      theme === 'dark' ? 'bg-gray-900/50' : 'bg-white/80'
    } backdrop-blur-sm overflow-hidden transition-all duration-300 hover:${colors.glow}`}>
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className={`text-3xl ${colors.text} mb-2`}>
              {algorithm}
            </h2>
            <Badge className={`${colors.bg} ${colors.text} border-none`}>
              Detected Clusters: {numDetectedClusters}
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Controls */}
          <div className="space-y-6">
            {/* Algorithm Insight */}
            <div className={`p-4 rounded-lg border-2 ${colors.border} ${
              theme === 'dark' ? 'bg-gradient-to-br from-gray-900/80 to-gray-800/50' : 'bg-gradient-to-br from-white to-gray-50'
            }`}>
              <h3 className={`text-base mb-2 ${colors.text}`}>How it works:</h3>
              {algorithm === 'BIRCH' && (
                <p className={`text-sm mb-3 pb-3 border-b ${theme === 'dark' ? 'border-gray-700 text-pink-300' : 'border-pink-200 text-pink-700'}`}>
                  BIRCH bridges the gap — combining K-Means' speed with DBSCAN's robustness, making it ideal for big, evolving datasets.
                </p>
              )}
              <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>
                {algorithm === 'K-Means' && (
                  <>
                    <span className={`${colors.text} block mb-1`}>"Every point belongs somewhere, even if it doesn't really fit."</span>
                    <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
                      Forced assignment - all points must join a cluster, regardless of how far they are from centers.
                    </span>
                  </>
                )}
                {algorithm === 'DBSCAN' && (
                  <>
                    <span className={`${colors.text} block mb-1`}>"Only dense groups matter, lonely points are ignored as noise."</span>
                    <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
                      Explicit noise detection - isolated points that don't fit dense regions are marked as outliers.
                    </span>
                  </>
                )}
                {algorithm === 'BIRCH' && (
                  <>
                    <span className={`${colors.text} block mb-1`}>"Summarize points into mini-buckets; lonely points form tiny buckets."</span>
                    <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
                      CF subclusters isolate noise - outliers create small subclusters that don't distort main clusters.
                    </span>
                  </>
                )}
              </p>
            </div>

            {algorithm === 'K-Means' && (
              <div className="space-y-2">
                <Label className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                  Number of Clusters: <span className={colors.text}>{numClusters}</span>
                </Label>
                <Slider
                  value={[numClusters || 3]}
                  onValueChange={(value) => onNumClustersChange?.(value[0])}
                  min={2}
                  max={8}
                  step={1}
                  className="cursor-pointer"
                />
              </div>
            )}

            {algorithm === 'DBSCAN' && (
              <>
                <div className="space-y-2">
                  <Label className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                    Epsilon (eps): <span className={colors.text}>{eps?.toFixed(2)}</span>
                  </Label>
                  <Slider
                    value={[eps || 0.6]}
                    onValueChange={(value) => onEpsChange?.(value[0])}
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    className="cursor-pointer"
                  />
                </div>
                <div className="space-y-2">
                  <Label className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                    Min Samples: <span className={colors.text}>{minSamples}</span>
                  </Label>
                  <Slider
                    value={[minSamples || 5]}
                    onValueChange={(value) => onMinSamplesChange?.(value[0])}
                    min={2}
                    max={20}
                    step={1}
                    className="cursor-pointer"
                  />
                </div>
              </>
            )}

            {algorithm === 'BIRCH' && (
              <div className="space-y-2">
                <Label className={theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}>
                  Number of Clusters: <span className={colors.text}>{numClusters}</span>
                </Label>
                <Slider
                  value={[numClusters || 3]}
                  onValueChange={(value) => onNumClustersChange?.(value[0])}
                  min={2}
                  max={8}
                  step={1}
                  className="cursor-pointer"
                />
              </div>
            )}

            {/* Legend */}
            <div className={`p-4 rounded-lg ${colors.bg} border ${colors.border}`}>
              <h3 className={`text-sm mb-3 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>Legend</h3>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"></div>
                  <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Cluster Points</span>
                </div>
                {algorithm === 'DBSCAN' && (
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${theme === 'dark' ? 'bg-gray-500' : 'bg-gray-400'}`}></div>
                    <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Noise Points</span>
                  </div>
                )}
                {algorithm === 'BIRCH' && (
                  <div className="flex items-center gap-2">
                    <div className="text-red-500">✕</div>
                    <span className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>Subcluster Centers</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Scatter Plot */}
          <div className="lg:col-span-3">
            <ScatterPlot
              data={data}
              centers={centers}
              algorithm={algorithm}
              color={color}
              theme={theme}
            />
          </div>
        </div>
      </div>
    </Card>
  );
}