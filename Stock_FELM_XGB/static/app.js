// React and Recharts objects are already available globally from index.html
const { useState, useEffect, useRef } = React;
const { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart } = Recharts;

const API_BASE = 'http://localhost:5000/api';

const IntegratedTradingDashboard = () => {
  const [stockData, setStockData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [status, setStatus] = useState(null);
  const [symbol, setSymbol] = useState('ZOMATO.NS');
  const [apiKey, setApiKey] = useState('');
  const [showConfig, setShowConfig] = useState(true);
  const [countdown, setCountdown] = useState(600);

  const intervalRef = useRef(null);
  const statusIntervalRef = useRef(null);
  const countdownRef = useRef(null);

  // Fetch status from backend
  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();
      setStatus(data);
      if (data.is_live && !countdownRef.current) {
        startCountdown();
      } else if (!data.is_live && countdownRef.current) {
        stopCountdown();
      }
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  // Fetch real-time data
  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE}/data`);
      const result = await response.json();
      if (result.data) setStockData(result.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  // Fetch predictions
  const fetchPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE}/predictions`);
      const result = await response.json();
      if (result.predictions) setPredictions(result.predictions);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  // Start live trading
  const startTrading = async () => {
    try {
      const response = await fetch(`${API_BASE}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, api_key: apiKey })
      });
      if (response.ok) {
        setShowConfig(false);
        intervalRef.current = setInterval(() => {
          fetchData();
          fetchPredictions();
        }, 5000);
        statusIntervalRef.current = setInterval(fetchStatus, 2000);
        startCountdown();
      }
    } catch (error) {
      console.error('Error starting trading:', error);
    }
  };

  // Stop live trading
  const stopTrading = async () => {
    try {
      await fetch(`${API_BASE}/stop`, { method: 'POST' });
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
      stopCountdown();
    } catch (error) {
      console.error('Error stopping trading:', error);
    }
  };

  // Countdown
  const startCountdown = () => {
    setCountdown(600);
    countdownRef.current = setInterval(() => {
      setCountdown(prev => (prev <= 1 ? 600 : prev - 1));
    }, 1000);
  };
  const stopCountdown = () => {
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
      countdownRef.current = null;
    }
  };

  // Manual train trigger
  const triggerTraining = async () => {
    try {
      await fetch(`${API_BASE}/train`, { method: 'POST' });
    } catch (error) {
      console.error('Error triggering training:', error);
    }
  };

  useEffect(() => {
    fetchStatus();
    const initialInterval = setInterval(fetchStatus, 5000);
    return () => {
      clearInterval(initialInterval);
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (statusIntervalRef.current) clearInterval(statusIntervalRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  }, []);

  const formatCountdown = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const chartData = [
    ...stockData.map(d => ({ time: d.timestamp, actual: d.close })),
    ...(predictions.length > 0
      ? [{
          time: stockData.length > 0 ? stockData[stockData.length - 1].timestamp : '',
          actual: stockData.length > 0 ? stockData[stockData.length - 1].close : null,
          predicted: stockData.length > 0 ? stockData[stockData.length - 1].close : null
        }]
      : []),
    ...predictions.map(p => ({ time: p.timestamp, predicted: p.price }))
  ];

  const currentPrice = stockData.length > 0 ? stockData[stockData.length - 1].close : null;
  const priceChange =
    stockData.length >= 2
      ? ((stockData[stockData.length - 1].close - stockData[stockData.length - 2].close) /
          stockData[stockData.length - 2].close) *
        100
      : 0;

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-lg">
          <p className="text-slate-300 text-sm mb-1">{payload[0].payload.time}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="font-semibold">
              {entry.name}: ₹{entry.value?.toFixed(2)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
          Real-Time Trading Intelligence
        </h1>

        <p className="text-slate-400 mb-6">XGBoost-Powered Intraday Predictions</p>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-6 border border-slate-700 shadow-lg mb-6">
          <h2 className="text-xl font-semibold mb-4">Live Chart</h2>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
              <XAxis dataKey="time" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="actual" stroke="#06b6d4" strokeWidth={3} fill="url(#colorActual)" name="Actual" />
              <Area type="monotone" dataKey="predicted" stroke="#a855f7" strokeWidth={3} strokeDasharray="8 4" fill="url(#colorPredicted)" name="Predicted" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// ✅ Render the component
ReactDOM.render(<IntegratedTradingDashboard />, document.getElementById('root'));
