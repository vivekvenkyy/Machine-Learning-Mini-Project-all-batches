import { useState, useEffect } from 'react';
import { ClusteringPanel } from './ClusteringPanel';
import { generateDataset, addOutliers } from './clustering-utils';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { ScatterPlot } from './ScatterPlot';
import { Plus, Sparkles, Sun, Moon, Eye } from 'lucide-react';

export interface DataPoint {
  x: number;
  y: number;
  cluster: number;
  isOutlier?: boolean;
}

interface ClusteringDashboardProps {
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

export function ClusteringDashboard({ theme, setTheme }: ClusteringDashboardProps) {
  const [baseData, setBaseData] = useState<DataPoint[]>([]);
  const [kMeansData, setKMeansData] = useState<DataPoint[]>([]);
  const [dbscanData, setDBScanData] = useState<DataPoint[]>([]);
  const [birchData, setBirchData] = useState<DataPoint[]>([]);
  
  const [kMeansClusters, setKMeansClusters] = useState(3);
  const [dbscanEps, setDBScanEps] = useState(0.6);
  const [dbscanMinSamples, setDBScanMinSamples] = useState(5);
  const [birchClusters, setBirchClusters] = useState(3);
  
  const [datasetType, setDatasetType] = useState('blobs');
  const [showDatasetDialog, setShowDatasetDialog] = useState(false);

  // Initialize data
  useEffect(() => {
    const initialData = generateDataset(datasetType, 300);
    setBaseData(initialData);
    setKMeansData([...initialData]);
    setDBScanData([...initialData]);
    setBirchData([...initialData]);
  }, [datasetType]);

  const handleAddOutliers = () => {
    const newOutliers = addOutliers(15);
    setBaseData(prev => [...prev, ...newOutliers]);
    setKMeansData(prev => [...prev, ...newOutliers]);
    setDBScanData(prev => [...prev, ...newOutliers]);
    setBirchData(prev => [...prev, ...newOutliers]);
  };

  const handleReset = () => {
    const initialData = generateDataset(datasetType, 300);
    setBaseData(initialData);
    setKMeansData([...initialData]);
    setDBScanData([...initialData]);
    setBirchData([...initialData]);
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-end mb-4">
            <Button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              variant="outline"
              size="icon"
              className={`${
                theme === 'dark'
                  ? 'border-cyan-500/50 text-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-200'
                  : 'border-purple-300 text-purple-600 hover:bg-purple-100 hover:text-purple-700'
              }`}
            >
              {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </Button>
          </div>
          <h1 className={`text-5xl mb-4 ${
            theme === 'dark'
              ? 'bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent'
              : 'bg-gradient-to-r from-cyan-600 via-purple-600 to-pink-600 bg-clip-text text-transparent'
          }`}>
            Clustering Simulation Dashboard
          </h1>
          <p className={`text-lg ${theme === 'dark' ? 'text-cyan-300/70' : 'text-slate-600'}`}>
            Interactive visualization of K-Means, DBSCAN, and BIRCH clustering algorithms
          </p>
          
          {/* Global Controls */}
          <div className="flex flex-col items-center gap-4 mt-6">
            <div className="flex items-center gap-3">
              <label className={`${theme === 'dark' ? 'text-cyan-300' : 'text-slate-700'}`}>
                Dataset:
              </label>
              <Select value={datasetType} onValueChange={setDatasetType}>
                <SelectTrigger 
                  className={`w-48 ${
                    theme === 'dark'
                      ? 'border-cyan-500/50 bg-slate-900/50 text-cyan-300 hover:bg-cyan-500/10'
                      : 'border-purple-300 bg-white text-purple-700 hover:bg-purple-50'
                  }`}
                >
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent className={theme === 'dark' ? 'bg-slate-900 border-cyan-500/50' : 'bg-white border-purple-300'}>
                  <SelectItem value="blobs" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Blobs
                  </SelectItem>
                  <SelectItem value="circles" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Circles
                  </SelectItem>
                  <SelectItem value="moons" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Moons
                  </SelectItem>
                  <SelectItem value="spirals" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Spirals
                  </SelectItem>
                  <SelectItem value="grid" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Grid
                  </SelectItem>
                  <SelectItem value="random" className={theme === 'dark' ? 'text-cyan-300 focus:bg-cyan-500/20' : 'text-purple-700 focus:bg-purple-100'}>
                    Random
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex gap-4">
              <Dialog open={showDatasetDialog} onOpenChange={setShowDatasetDialog}>
                <DialogTrigger asChild>
                  <Button
                    variant="outline"
                    className={`${
                      theme === 'dark'
                        ? 'border-cyan-500/50 text-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-200'
                        : 'border-purple-300 text-purple-600 hover:bg-purple-100 hover:text-purple-700'
                    }`}
                  >
                    <Eye className="mr-2 h-4 w-4" />
                    View Raw Dataset
                  </Button>
                </DialogTrigger>
                <DialogContent className={`max-w-3xl ${
                  theme === 'dark'
                    ? 'bg-slate-900 border-cyan-500/50'
                    : 'bg-white border-purple-300'
                }`}>
                  <DialogHeader>
                    <DialogTitle className={theme === 'dark' ? 'text-cyan-300' : 'text-purple-700'}>
                      Raw Dataset Preview
                    </DialogTitle>
                    <DialogDescription className={theme === 'dark' ? 'text-cyan-300/70' : 'text-slate-600'}>
                      Original data structure before clustering algorithms are applied. Points are colored by their initial grouping.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="mt-4">
                    <ScatterPlot
                      data={baseData}
                      width={700}
                      height={500}
                      color="cyan"
                      theme={theme}
                      showCentroids={false}
                    />
                  </div>
                </DialogContent>
              </Dialog>
              
              <Button
                onClick={handleAddOutliers}
                className={`${
                  theme === 'dark'
                    ? 'bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 text-white shadow-lg shadow-pink-500/50'
                    : 'bg-gradient-to-r from-pink-400 to-purple-500 hover:from-pink-500 hover:to-purple-600 text-white shadow-lg shadow-pink-400/30'
                } transition-all`}
              >
                <Plus className="mr-2 h-4 w-4" />
                Add Random Outliers
              </Button>
              <Button
                onClick={handleReset}
                variant="outline"
                className={`${
                  theme === 'dark'
                    ? 'border-cyan-500/50 text-cyan-300 hover:bg-cyan-500/10 hover:text-cyan-200'
                    : 'border-purple-300 text-purple-600 hover:bg-purple-100 hover:text-purple-700'
                }`}
              >
                <Sparkles className="mr-2 h-4 w-4" />
                Reset Data
              </Button>
            </div>
          </div>
        </div>

        {/* Clustering Panels */}
        <div className="space-y-8">
          <ClusteringPanel
            algorithm="K-Means"
            data={kMeansData}
            setData={setKMeansData}
            baseData={baseData}
            numClusters={kMeansClusters}
            onNumClustersChange={setKMeansClusters}
            color="cyan"
            theme={theme}
          />
          
          <ClusteringPanel
            algorithm="DBSCAN"
            data={dbscanData}
            setData={setDBScanData}
            baseData={baseData}
            eps={dbscanEps}
            minSamples={dbscanMinSamples}
            onEpsChange={setDBScanEps}
            onMinSamplesChange={setDBScanMinSamples}
            color="purple"
            theme={theme}
          />
          
          <ClusteringPanel
            algorithm="BIRCH"
            data={birchData}
            setData={setBirchData}
            baseData={baseData}
            numClusters={birchClusters}
            onNumClustersChange={setBirchClusters}
            color="pink"
            theme={theme}
          />
        </div>
      </div>
    </div>
  );
}