import React, { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Legend, ZAxis, Label } from 'recharts';
import { Point } from '../types';

interface ClusterPlotProps {
  data: Point[];
  title: string;
  xAxisLabel: string;
  yAxisLabel: string;
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088FE', '#00C49F', '#FFBB28', '#FF8042'];
const NOISE_COLOR = '#6b7280';

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-gray-800 p-2 border border-gray-600 rounded-md text-sm">
        <p className="label">{`Point: (${data.x.toFixed(2)}, ${data.y.toFixed(2)})`}</p>
        <p className="intro">{`Cluster: ${data.cluster === -1 ? 'Noise' : data.cluster}`}</p>
      </div>
    );
  }

  return null;
};

const ClusterPlot: React.FC<ClusterPlotProps> = ({ data, title, xAxisLabel, yAxisLabel }) => {
    // Group data by cluster for rendering separate scatter plots (which enables the legend)
    const dataByCluster = useMemo(() => {
        return data.reduce((acc, point) => {
            const augmentedPoint = { ...point, z: 1 }; // Add dummy 'z' property for size control
            const clusterId = point.cluster ?? -1;
            if (!acc[clusterId]) {
                acc[clusterId] = [];
            }
            acc[clusterId].push(augmentedPoint);
            return acc;
        }, {} as Record<number, (Point & { z: number })[]>);
    }, [data]);

  return (
    <div className="bg-gray-800/50 p-4 rounded-xl shadow-lg w-full h-96 flex flex-col">
      <h3 className="text-lg font-bold text-center mb-2 text-violet-300">{title}</h3>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 20 }}>
            <XAxis type="number" dataKey="x" name={xAxisLabel} tick={{ fill: '#9ca3af' }} axisLine={{ stroke: '#4b5563' }} tickLine={{ stroke: '#4b5563' }}>
                <Label value={xAxisLabel} offset={-20} position="insideBottom" fill="#9ca3af" />
            </XAxis>
            <YAxis type="number" dataKey="y" name={yAxisLabel} tick={{ fill: '#9ca3af' }} axisLine={{ stroke: '#4b5563' }} tickLine={{ stroke: '#4b5563' }} >
                <Label value={yAxisLabel} angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#9ca3af' }} />
            </YAxis>
            {/* ZAxis is used to control the visual size of the dots in the scatter plot */}
            <ZAxis dataKey="z" range={[40, 40]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Legend wrapperStyle={{paddingTop: '25px'}} />
            {Object.entries(dataByCluster).map(([clusterIdStr, points]) => {
                const clusterId = Number(clusterIdStr);
                return (
                    <Scatter
                        key={clusterId}
                        name={clusterId === -1 ? 'Noise' : `Cluster ${clusterId}`}
                        data={points}
                        fill={clusterId === -1 ? NOISE_COLOR : COLORS[clusterId % COLORS.length]}
                        shape="circle"
                    />
                );
            })}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ClusterPlot;