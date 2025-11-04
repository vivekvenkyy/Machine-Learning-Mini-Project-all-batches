import React, { useState, useRef, useEffect } from 'react';
import { DatasetType, AlgorithmParameters, LinkageType } from '../types';
import { UploadIcon, AnalyzeIcon } from './IconComponents';

interface ControlPanelProps {
  onAnalyze: (datasetType: DatasetType, file: File | null, params: AlgorithmParameters, dropLastColumn: boolean) => void;
  isLoading: boolean;
}

const getHintForDataset = (datasetType: DatasetType): string => {
    switch(datasetType) {
        case DatasetType.Moons:
        case DatasetType.Circles:
            return "Tests how algorithms handle non-convex shapes.";
        case DatasetType.Blobs:
            return "Ideal for standard, distance-based clustering.";
        default:
            return "";
    }
};

const ControlPanel: React.FC<ControlPanelProps> = ({ onAnalyze, isLoading }) => {
  const [datasetType, setDatasetType] = useState<DatasetType>(DatasetType.Moons);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dropLastColumn, setDropLastColumn] = useState(true);
  const [datasetHint, setDatasetHint] = useState<string>('');

  useEffect(() => {
    setDatasetHint(getHintForDataset(DatasetType.Moons));
  }, []);

  const [params, setParams] = useState<AlgorithmParameters>({
    spectral: { n_clusters: 2 },
    kmeans: { n_clusters: 2 },
    agglomerative: { n_clusters: 2, linkage: 'ward' },
    dbscan: { eps: 0.5 },
  });

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
      setDatasetType(DatasetType.Custom);
      setDatasetHint('');
    }
  };

  const handleAnalyzeClick = () => {
    onAnalyze(datasetType, selectedFile, params, dropLastColumn);
  };
  
  const handleDatasetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = e.target.value as DatasetType;
    setDatasetType(newType);
    setDatasetHint(getHintForDataset(newType));
    
    if (newType !== DatasetType.Custom) {
        setSelectedFile(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }
    // Reset clusters based on dataset
    let newClusters = 2;
    if (newType === DatasetType.Blobs) newClusters = 3;
    
    setParams(prev => ({
        ...prev,
        spectral: { n_clusters: newClusters },
        kmeans: { n_clusters: newClusters },
        agglomerative: { ...prev.agglomerative, n_clusters: newClusters },
    }))
  };

  return (
    <div className="p-6 bg-gray-900/70 backdrop-blur-sm border border-gray-700 rounded-2xl shadow-2xl h-full flex flex-col space-y-6">
      <h2 className="text-2xl font-bold text-center text-violet-400">Controls</h2>

      {/* Dataset Selection */}
      <div>
        <label htmlFor="dataset" className="block text-sm font-medium text-gray-300 mb-2">
          1. Select a Dataset
        </label>
        <select
          id="dataset"
          value={datasetType}
          onChange={handleDatasetChange}
          className="w-full bg-gray-800 border border-gray-600 rounded-lg py-2 px-3 text-white focus:ring-2 focus:ring-violet-500 focus:border-violet-500"
        >
          <option value={DatasetType.Moons}>Moons</option>
          <option value={DatasetType.Blobs}>Blobs</option>
          <option value={DatasetType.Circles}>Circles</option>
          <option value={DatasetType.Custom} disabled>Custom (upload below)</option>
        </select>
        {datasetHint && <p className="text-xs text-gray-500 mt-2 px-1 italic">Hint: {datasetHint}</p>}
      </div>

      {/* File Upload */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Or Upload Your Own (CSV)
        </label>
        <label
          htmlFor="file-upload"
          className={`relative cursor-pointer bg-gray-800 rounded-md font-medium ${
            isLoading ? 'opacity-50 cursor-not-allowed' : 'text-violet-400 hover:text-violet-300'
          } focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-offset-gray-900 focus-within:ring-violet-500 border border-gray-600 p-3 flex justify-center items-center`}
        >
          <UploadIcon className="w-5 h-5 mr-2" />
          <span>{selectedFile ? selectedFile.name : 'Choose a file...'}</span>
          <input 
            ref={fileInputRef} 
            id="file-upload" 
            name="file-upload" 
            type="file" 
            className="sr-only" 
            accept=".csv" 
            onChange={handleFileChange}
            disabled={isLoading}
          />
        </label>
        <p className="text-xs text-gray-500 mt-2 px-1 italic">
          Supported format: CSV with numeric data, max 10MB. First row should be headers.
        </p>
        {selectedFile && (
          <div className="space-y-3 mt-3">
            <div className="flex items-center">
              <input
                id="drop-last-col"
                type="checkbox"
                checked={dropLastColumn}
                onChange={(e) => setDropLastColumn(e.target.checked)}
                className="h-4 w-4 rounded border-gray-500 bg-gray-700 text-violet-500 focus:ring-violet-600"
                disabled={isLoading}
              />
              <label htmlFor="drop-last-col" className="ml-2 block text-sm text-gray-300">
                Drop last column (target variable)
              </label>
            </div>
            <button
              onClick={() => {
                setSelectedFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
                setDatasetType(DatasetType.Moons);
              }}
              disabled={isLoading}
              className="text-sm text-red-400 hover:text-red-300 disabled:opacity-50"
            >
              Remove file
            </button>
          </div>
        )}
      </div>

      {/* Algorithm Parameters */}
      <div>
        <h3 className="block text-sm font-medium text-gray-300 mb-3">2. Adjust Parameters</h3>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 items-center">
            <label htmlFor="n_clusters" className="text-gray-400">Clusters (K)</label>
            <input
              type="number"
              id="n_clusters"
              min="2"
              max="8"
              value={params.kmeans.n_clusters}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (isNaN(val) || val < 2) return;
                setParams(prev => ({
                  ...prev,
                  spectral: { n_clusters: val },
                  kmeans: { n_clusters: val },
                  agglomerative: { ...prev.agglomerative, n_clusters: val },
                }));
              }}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg py-1 px-2 text-white"
            />
          </div>
          <div className="grid grid-cols-2 gap-4 items-center">
            <label htmlFor="linkage" className="text-gray-400">Agglo. Linkage</label>
            <select
              id="linkage"
              value={params.agglomerative.linkage}
              onChange={(e) => {
                const val = e.target.value as LinkageType;
                setParams(prev => ({
                    ...prev,
                    agglomerative: { ...prev.agglomerative, linkage: val }
                }));
              }}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg py-1 px-2 text-white"
            >
                <option value="ward">ward</option>
                <option value="complete">complete</option>
                <option value="average">average</option>
                <option value="single">single</option>
            </select>
          </div>
          <div className="grid grid-cols-2 gap-4 items-center">
            <label htmlFor="dbscan_eps" className="text-gray-400">DBSCAN Epsilon</label>
            <input
              type="number"
              id="dbscan_eps"
              min="0.1"
              max="2.0"
              step="0.1"
              value={params.dbscan.eps}
              onChange={(e) => setParams({ ...params, dbscan: { eps: parseFloat(e.target.value) } })}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg py-1 px-2 text-white"
            />
          </div>
        </div>
      </div>
      
      <div className="flex-grow"></div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyzeClick}
        disabled={isLoading}
        className="w-full flex justify-center items-center py-3 px-4 bg-violet-600 hover:bg-violet-700 text-white font-bold rounded-lg shadow-lg transition-all duration-300 disabled:bg-gray-500 disabled:cursor-not-allowed transform hover:scale-105"
      >
        <AnalyzeIcon className="w-5 h-5 mr-2"/>
        {isLoading ? 'Analyzing...' : 'Visualize & Analyze'}
      </button>
    </div>
  );
};

export default ControlPanel;