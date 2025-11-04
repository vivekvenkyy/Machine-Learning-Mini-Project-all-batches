import { useState } from 'react';
import { ClusteringDashboard } from './components/ClusteringDashboard';

export default function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  return (
    <div className={`min-h-screen ${theme === 'dark' ? 'bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50'}`}>
      <ClusteringDashboard theme={theme} setTheme={setTheme} />
    </div>
  );
}