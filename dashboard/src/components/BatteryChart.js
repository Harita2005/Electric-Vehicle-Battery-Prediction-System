import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine
} from 'recharts';
import { format } from 'date-fns';

const BatteryChart = ({ data, type = 'timeline', height = 400 }) => {
  const formatTooltipLabel = (label) => {
    if (label instanceof Date) {
      return format(label, 'MMM dd, yyyy HH:mm');
    }
    return label;
  };

  const formatXAxisLabel = (tickItem) => {
    if (tickItem instanceof Date) {
      return format(tickItem, 'MMM dd');
    }
    return tickItem;
  };

  if (type === 'scatter') {
    // Fleet health distribution scatter plot
    return (
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="soh" 
            name="SoH (%)"
            domain={['dataMin - 5', 'dataMax + 5']}
          />
          <YAxis 
            dataKey="rul" 
            name="RUL (days)"
            domain={['dataMin - 50', 'dataMax + 50']}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value, name) => [value, name === 'soh' ? 'SoH (%)' : 'RUL (days)']}
            labelFormatter={(label, payload) => 
              payload && payload[0] ? `Vehicle: ${payload[0].payload.vehicle_id}` : ''
            }
          />
          <Scatter dataKey="rul" fill="#1976d2" />
          <ReferenceLine x={80} stroke="red" strokeDasharray="5 5" />
          <ReferenceLine y={365} stroke="orange" strokeDasharray="5 5" />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'timeline') {
    // SoH timeline chart
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp"
            tickFormatter={formatXAxisLabel}
            type="category"
            scale="time"
          />
          <YAxis 
            domain={['dataMin - 2', 'dataMax + 2']}
            label={{ value: 'SoH (%)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            labelFormatter={formatTooltipLabel}
            formatter={(value) => [value.toFixed(2), 'SoH (%)']}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="soh" 
            stroke="#1976d2" 
            strokeWidth={2}
            dot={false}
            name="State of Health"
          />
          <ReferenceLine y={80} stroke="red" strokeDasharray="5 5" label="EOL Threshold" />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'voltage_current') {
    // Voltage and current chart
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp"
            tickFormatter={formatXAxisLabel}
          />
          <YAxis yAxisId="voltage" orientation="left" />
          <YAxis yAxisId="current" orientation="right" />
          <Tooltip 
            labelFormatter={formatTooltipLabel}
            formatter={(value, name) => [
              value.toFixed(1), 
              name === 'pack_voltage' ? 'Voltage (V)' : 'Current (A)'
            ]}
          />
          <Legend />
          <Line 
            yAxisId="voltage"
            type="monotone" 
            dataKey="pack_voltage" 
            stroke="#1976d2" 
            strokeWidth={1}
            dot={false}
            name="Pack Voltage (V)"
          />
          <Line 
            yAxisId="current"
            type="monotone" 
            dataKey="pack_current" 
            stroke="#dc004e" 
            strokeWidth={1}
            dot={false}
            name="Pack Current (A)"
          />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === 'temp_soc') {
    // Temperature and SOC chart
    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp"
            tickFormatter={formatXAxisLabel}
          />
          <YAxis yAxisId="temp" orientation="left" />
          <YAxis yAxisId="soc" orientation="right" domain={[0, 100]} />
          <Tooltip 
            labelFormatter={formatTooltipLabel}
            formatter={(value, name) => [
              value.toFixed(1), 
              name === 'pack_temp' ? 'Temperature (°C)' : 'SOC (%)'
            ]}
          />
          <Legend />
          <Line 
            yAxisId="temp"
            type="monotone" 
            dataKey="pack_temp" 
            stroke="#ff9800" 
            strokeWidth={1}
            dot={false}
            name="Pack Temperature (°C)"
          />
          <Line 
            yAxisId="soc"
            type="monotone" 
            dataKey="soc" 
            stroke="#4caf50" 
            strokeWidth={1}
            dot={false}
            name="State of Charge (%)"
          />
          <ReferenceLine yAxisId="temp" y={40} stroke="red" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return null;
};

export default BatteryChart;