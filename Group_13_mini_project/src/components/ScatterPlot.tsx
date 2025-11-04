import { useEffect, useRef, useState } from 'react';
import { DataPoint } from './ClusteringDashboard';
import { motion } from 'motion/react';

interface ScatterPlotProps {
  data: DataPoint[];
  centers?: { x: number; y: number }[];
  algorithm?: string;
  color: 'cyan' | 'purple' | 'pink';
  theme: 'light' | 'dark';
  width?: number;
  height?: number;
  showCentroids?: boolean;
}

const clusterColors = {
  dark: [
    '#06b6d4', // cyan-500
    '#8b5cf6', // violet-500
    '#ec4899', // pink-500
    '#f59e0b', // amber-500
    '#10b981', // emerald-500
    '#3b82f6', // blue-500
    '#f97316', // orange-500
    '#6366f1', // indigo-500
  ],
  light: [
    '#0891b2', // cyan-600
    '#7c3aed', // violet-600
    '#db2777', // pink-600
    '#d97706', // amber-600
    '#059669', // emerald-600
    '#2563eb', // blue-600
    '#ea580c', // orange-600
    '#4f46e5', // indigo-600
  ],
};

export function ScatterPlot({ 
  data, 
  centers = [], 
  algorithm = '', 
  color, 
  theme,
  width,
  height,
  showCentroids = true 
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; data: DataPoint } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const animationFrameRef = useRef<number>();
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const padding = 40;
    const plotWidth = rect.width - 2 * padding;
    const plotHeight = rect.height - 2 * padding;

    // Find data bounds
    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const centerXValues = centers.length > 0 ? centers.map(c => c.x) : [];
    const centerYValues = centers.length > 0 ? centers.map(c => c.y) : [];
    const xMin = Math.min(...xValues, ...centerXValues) - 0.5;
    const xMax = Math.max(...xValues, ...centerXValues) + 0.5;
    const yMin = Math.min(...yValues, ...centerYValues) - 0.5;
    const yMax = Math.max(...yValues, ...centerYValues) + 0.5;

    const scaleX = (x: number) => padding + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const scaleY = (y: number) => rect.height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight;

    const animate = () => {
      timeRef.current += 0.02;
      ctx.clearRect(0, 0, rect.width, rect.height);

      // Draw grid
      ctx.strokeStyle = theme === 'dark' ? 'rgba(100, 100, 100, 0.2)' : 'rgba(200, 200, 200, 0.3)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const x = padding + (i / 10) * plotWidth;
        const y = rect.height - padding - (i / 10) * plotHeight;
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, rect.height - padding);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(rect.width - padding, y);
        ctx.stroke();
      }

      // Draw axes
      ctx.strokeStyle = theme === 'dark' ? 'rgba(200, 200, 200, 0.5)' : 'rgba(100, 100, 100, 0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding, rect.height - padding);
      ctx.lineTo(rect.width - padding, rect.height - padding);
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, rect.height - padding);
      ctx.stroke();

      // Draw axis labels
      ctx.fillStyle = theme === 'dark' ? 'rgba(200, 200, 200, 0.7)' : 'rgba(60, 60, 60, 0.8)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1', rect.width / 2, rect.height - 10);
      ctx.save();
      ctx.translate(15, rect.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Feature 2', 0, 0);
      ctx.restore();

      // Draw data points with animation
      const colors = clusterColors[theme];
      data.forEach((point, idx) => {
        const x = scaleX(point.x);
        const y = scaleY(point.y);
        
        // Pulsing animation
        const pulse = 1 + 0.1 * Math.sin(timeRef.current * 2 + idx * 0.1);
        const radius = 4 * pulse;

        // Noise points (cluster -1) in DBSCAN
        if (algorithm === 'DBSCAN' && point.cluster === -1) {
          // Gray noise points
          ctx.fillStyle = theme === 'dark' ? 'rgba(156, 163, 175, 0.6)' : 'rgba(107, 114, 128, 0.7)';
          ctx.shadowBlur = theme === 'dark' ? 8 : 4;
          ctx.shadowColor = theme === 'dark' ? 'rgba(156, 163, 175, 0.8)' : 'rgba(107, 114, 128, 0.5)';
        } else {
          // Regular cluster points with glow
          const clusterColor = colors[point.cluster % colors.length];
          ctx.fillStyle = clusterColor;
          ctx.shadowBlur = theme === 'dark' 
            ? 15 + 5 * Math.sin(timeRef.current + idx * 0.1)
            : 8 + 3 * Math.sin(timeRef.current + idx * 0.1);
          ctx.shadowColor = clusterColor;
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Draw centers (K-Means) or subcluster centers (BIRCH)
      if (showCentroids && centers.length > 0) {
        centers.forEach((center, idx) => {
          const x = scaleX(center.x);
          const y = scaleY(center.y);

          if (algorithm === 'BIRCH') {
            // Red X markers with glow for BIRCH
            ctx.strokeStyle = theme === 'dark' ? '#ef4444' : '#dc2626';
            ctx.lineWidth = 3;
            ctx.shadowBlur = theme === 'dark' ? 20 : 10;
            ctx.shadowColor = theme === 'dark' ? '#ef4444' : '#dc2626';
            
            const size = 10;
            ctx.beginPath();
            ctx.moveTo(x - size, y - size);
            ctx.lineTo(x + size, y + size);
            ctx.moveTo(x + size, y - size);
            ctx.lineTo(x - size, y + size);
            ctx.stroke();
            ctx.shadowBlur = 0;
          } else {
            // Cluster centers for K-Means
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = colors[idx % colors.length];
            ctx.lineWidth = 3;
            ctx.shadowBlur = theme === 'dark' ? 20 : 10;
            ctx.shadowColor = colors[idx % colors.length];
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        });
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [data, centers, algorithm, theme]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    setMousePos({ x: e.clientX, y: e.clientY });

    const padding = 40;
    const plotWidth = rect.width - 2 * padding;
    const plotHeight = rect.height - 2 * padding;

    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const xMin = Math.min(...xValues) - 0.5;
    const xMax = Math.max(...xValues) + 0.5;
    const yMin = Math.min(...yValues) - 0.5;
    const yMax = Math.max(...yValues) + 0.5;

    const scaleX = (x: number) => padding + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const scaleY = (y: number) => rect.height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight;

    // Find closest point
    let closestPoint: DataPoint | null = null;
    let minDistance = Infinity;

    data.forEach(point => {
      const px = scaleX(point.x);
      const py = scaleY(point.y);
      const distance = Math.sqrt((px - mouseX) ** 2 + (py - mouseY) ** 2);
      
      if (distance < 10 && distance < minDistance) {
        minDistance = distance;
        closestPoint = point;
      }
    });

    if (closestPoint) {
      setHoveredPoint({
        x: scaleX(closestPoint.x),
        y: scaleY(closestPoint.y),
        data: closestPoint,
      });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="relative w-full" style={{ height: height ? `${height}px` : '500px' }}>
      <canvas
        ref={canvasRef}
        className={`w-full h-full rounded-lg ${
          theme === 'dark'
            ? 'bg-gradient-to-br from-gray-950 to-gray-900'
            : 'bg-gradient-to-br from-gray-50 to-white'
        }`}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: hoveredPoint ? 'pointer' : 'default' }}
      />
      
      {/* Tooltip */}
      {hoveredPoint && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed z-50 pointer-events-none"
          style={{
            left: mousePos.x + 10,
            top: mousePos.y + 10,
          }}
        >
          <div className={`${
            theme === 'dark'
              ? 'bg-gray-900/95 border-cyan-500/50 shadow-cyan-500/20'
              : 'bg-white/95 border-purple-300 shadow-purple-200/40'
          } border rounded-lg px-3 py-2 shadow-lg backdrop-blur-sm`}>
            <div className="text-xs space-y-1">
              <div className={theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}>
                X: {hoveredPoint.data.x.toFixed(3)}
              </div>
              <div className={theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}>
                Y: {hoveredPoint.data.y.toFixed(3)}
              </div>
              <div className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
                Cluster: {hoveredPoint.data.cluster === -1 ? 'Noise' : hoveredPoint.data.cluster}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}