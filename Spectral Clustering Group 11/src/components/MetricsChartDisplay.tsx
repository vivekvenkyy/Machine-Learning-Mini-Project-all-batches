import React from 'react';
import { ClusterResult, Metrics } from '../types';

interface MetricBarProps {
    label: string;
    value: number;
    maxValue: number;
    lowerIsBetter?: boolean;
}

const MetricBar: React.FC<MetricBarProps> = ({ label, value, maxValue, lowerIsBetter = false }) => {
    // For Silhouette, range is -1 to 1. We normalize to 0-1 for width calculation.
    const normalizedValue = label === 'Silhouette' ? (value + 1) / 2 : value;
    const normalizedMaxValue = label === 'Silhouette' ? 1 : maxValue;
    
    let widthPercent = (normalizedMaxValue > 0) ? (normalizedValue / normalizedMaxValue) * 100 : 0;
    if (widthPercent < 0) widthPercent = 0;
    if (widthPercent > 100) widthPercent = 100;

    let barColor = 'bg-cyan-500';
    if (lowerIsBetter) {
        if (widthPercent > 75) barColor = 'bg-red-500';
        else if (widthPercent > 40) barColor = 'bg-yellow-500';
        else barColor = 'bg-green-500';
    } else {
        if (widthPercent < 25) barColor = 'bg-red-500';
        else if (widthPercent < 60) barColor = 'bg-yellow-500';
        else barColor = 'bg-green-500';
    }

    return (
        <div className="grid grid-cols-3 items-center gap-2">
            <span className="text-gray-400 text-xs truncate">{label}</span>
            <div className="col-span-2 flex items-center">
                <div className="w-full bg-gray-700 rounded-full h-2.5 mr-2">
                    <div className={`${barColor} h-2.5 rounded-full`} style={{ width: `${widthPercent}%` }}></div>
                </div>
                <span className="font-mono text-cyan-300 text-xs w-12 text-right">{value.toFixed(2)}</span>
            </div>
        </div>
    );
};


interface MetricsChartDisplayProps {
    result: ClusterResult;
    maxMetrics: Metrics;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const MetricsChartDisplay: React.FC<MetricsChartDisplayProps> = ({ result, maxMetrics }) => {
    const formatParamKey = (key: string): string => {
        return key.replace('n_clusters', 'K').split('_').map(capitalize).join(' ');
    };

    return (
        <div className="bg-gray-800/60 p-3 rounded-b-xl text-xs space-y-3 mt-0">
            <div className="flex justify-start items-center space-x-4 border-b border-gray-700 pb-2 mb-2">
                 <span className="text-gray-300 font-bold">Params:</span>
                 <div className="flex space-x-3">
                    {Object.entries(result.params).map(([key, value]) => (
                        <div key={key}>
                            <span className="text-gray-400">{formatParamKey(key)}: </span>
                            <span className="font-mono text-violet-300">{value}</span>
                        </div>
                    ))}
                 </div>
            </div>
            <div className="space-y-2">
                <MetricBar label="Silhouette" value={result.metrics.silhouette} maxValue={1} />
                <MetricBar label="Calinski-Harabasz" value={result.metrics.calinskiHarabasz} maxValue={maxMetrics.calinskiHarabasz} />
                <MetricBar label="Davies-Bouldin" value={result.metrics.daviesBouldin} maxValue={maxMetrics.daviesBouldin} lowerIsBetter={true} />
            </div>
        </div>
    );
};

export default MetricsChartDisplay;
