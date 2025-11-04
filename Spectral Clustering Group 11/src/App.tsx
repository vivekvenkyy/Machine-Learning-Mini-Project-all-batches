import React, { useState, useEffect, useCallback } from 'react';
import ControlPanel from './components/ControlPanel';
import ClusterPlot from './components/ClusterPlot';
import MetricsChartDisplay from './components/MetricsChartDisplay';
import { LoaderIcon } from './components/IconComponents';
import { Point, ClusterResult, DatasetType, AlgorithmParameters, Metrics } from './types';
import {
  generateDataset,
  parseCSV,
  runSpectralClustering,
  runKMeans,
  runAgglomerativeClustering,
  runDBSCAN,
} from './services/clusteringService';
import { analyzeClusteringResults } from './services/geminiService';
import { marked } from 'marked';

const App: React.FC = () => {
    const [dataset, setDataset] = useState<Point[]>([]);
    const [datasetName, setDatasetName] = useState<string>('Moons');
    const [clusterResults, setClusterResults] = useState<ClusterResult[]>([]);
    const [maxMetrics, setMaxMetrics] = useState<Metrics | null>(null);
    const [geminiAnalysis, setGeminiAnalysis] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [xAxisLabel, setXAxisLabel] = useState('Feature 1');
    const [yAxisLabel, setYAxisLabel] = useState('Feature 2');


    const runAnalysis = useCallback(async (
        datasetType: DatasetType,
        file: File | null,
        params: AlgorithmParameters,
        dropLastColumn: boolean
    ) => {
        setIsLoading(true);
        setError(null);
        setGeminiAnalysis('');
        setClusterResults([]);
        setMaxMetrics(null);

        try {
            let currentData: Point[];
            let currentDatasetName: string = datasetType;

            if (datasetType === DatasetType.Custom && file) {
                currentData = await parseCSV(file, { dropLastColumn });
                currentDatasetName = file.name;
                setXAxisLabel('Principal Component 1');
                setYAxisLabel('Principal Component 2');
            } else {
                currentData = generateDataset(datasetType);
                setXAxisLabel('Feature 1');
                setYAxisLabel('Feature 2');
            }

            if (currentData.length === 0) {
                setError("Could not load or generate data. Please select a valid dataset or upload a valid CSV file.");
                setIsLoading(false);
                return;
            }

            setDataset(currentData);
            setDatasetName(currentDatasetName);

            const results: ClusterResult[] = [
                runSpectralClustering(currentData, params.spectral, datasetType),
                runKMeans(currentData, params.kmeans, datasetType),
                runAgglomerativeClustering(currentData, params.agglomerative, datasetType),
                runDBSCAN(currentData, params.dbscan, datasetType),
            ];
            
            // Calculate max metrics for chart normalization
            const maxes: Metrics = results.reduce((acc, result) => ({
                silhouette: Math.max(acc.silhouette, result.metrics.silhouette),
                calinskiHarabasz: Math.max(acc.calinskiHarabasz, result.metrics.calinskiHarabasz),
                daviesBouldin: Math.max(acc.daviesBouldin, result.metrics.daviesBouldin),
            }), { silhouette: -Infinity, calinskiHarabasz: -Infinity, daviesBouldin: -Infinity });
            setMaxMetrics(maxes);
            setClusterResults(results);

            const analysis = await analyzeClusteringResults(results, currentDatasetName);
            setGeminiAnalysis(analysis);

        } catch (e) {
            console.error(e);
            setError(e instanceof Error ? e.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, []);
    
    useEffect(() => {
        // Initial run on page load
        runAnalysis(DatasetType.Moons, null, {
            spectral: { n_clusters: 2 },
            kmeans: { n_clusters: 2 },
            agglomerative: { n_clusters: 2, linkage: 'ward' },
            dbscan: { eps: 0.5 },
        }, false);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const parsedHtml = geminiAnalysis ? marked.parse(geminiAnalysis) : '';

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 font-sans p-4 lg:p-8">
            <div className="container mx-auto max-w-screen-2xl">
                <header className="text-center mb-8">
                    <h1 className="text-4xl lg:text-5xl font-extrabold">
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-purple-500">
                            Spectral Clustering Visualizer
                        </span>
                    </h1>
                    <p className="mt-2 text-lg text-gray-400">Compare clustering algorithms with AI-powered analysis</p>
                </header>

                <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <aside className="lg:col-span-3">
                        <div className="sticky top-8">
                            <ControlPanel onAnalyze={runAnalysis} isLoading={isLoading} />
                        </div>
                    </aside>

                    <div className="lg:col-span-9">
                        {error && (
                            <div className="bg-red-900/50 border border-red-700 text-red-300 p-4 rounded-lg mb-6">
                                <strong>Error:</strong> {error}
                            </div>
                        )}
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            {clusterResults.length > 0 && maxMetrics ? (
                                clusterResults.map(result => (
                                    <div key={result.algorithm}>
                                        <ClusterPlot 
                                            data={result.data} 
                                            title={result.algorithm}
                                            xAxisLabel={xAxisLabel}
                                            yAxisLabel={yAxisLabel}
                                        />
                                        <MetricsChartDisplay result={result} maxMetrics={maxMetrics} />
                                    </div>
                                ))
                            ) : isLoading ? (
                                Array.from({ length: 4 }).map((_, i) => (
                                    <div key={i} className="bg-gray-800/50 p-4 rounded-xl shadow-lg w-full h-[32rem] flex items-center justify-center">
                                        <LoaderIcon className="w-16 h-16"/>
                                    </div>
                                ))
                            ) : (
                                <div className="md:col-span-2 text-center py-16 text-gray-500">
                                    <p>Run analysis to see clustering visualizations.</p>
                                </div>
                            )}
                        </div>

                        <div className="bg-gray-900/70 backdrop-blur-sm border border-gray-700 rounded-2xl shadow-2xl p-6">
                            <h2 className="text-2xl font-bold mb-4 text-violet-400 flex items-center">
                                <span className="mr-2">✨</span> Gemini AI Analysis
                            </h2>
                            {isLoading && !geminiAnalysis ? (
                                <div className="flex flex-col items-center justify-center min-h-[200px]">
                                    <LoaderIcon className="w-12 h-12" />
                                    <p className="mt-4 text-gray-400">Gemini is analyzing the results...</p>
                                </div>
                            ) : geminiAnalysis ? (
                                <div className="prose prose-invert prose-p:text-gray-300 prose-headings:text-violet-300 prose-strong:text-white prose-li:text-gray-300" dangerouslySetInnerHTML={{ __html: parsedHtml }}/>
                            ) : (
                                <p className="text-gray-500">AI analysis will appear here after running the visualization.</p>
                            )}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
};

export default App;