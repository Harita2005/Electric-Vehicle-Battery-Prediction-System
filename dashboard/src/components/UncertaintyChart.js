import React from 'react';
import { Box, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';

const UncertaintyChart = ({ prediction, confidence_interval, uncertainty }) => {
  // Create data for uncertainty visualization
  const data = [
    {
      name: 'Lower Bound',
      value: confidence_interval[0],
      color: '#ff9800'
    },
    {
      name: 'Prediction',
      value: prediction,
      color: '#1976d2'
    },
    {
      name: 'Upper Bound',
      value: confidence_interval[1],
      color: '#ff9800'
    }
  ];

  const confidenceWidth = confidence_interval[1] - confidence_interval[0];
  const confidenceLevel = 90; // 90% confidence interval

  return (
    <Box>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="horizontal">
          <XAxis type="number" domain={['dataMin - 50', 'dataMax + 50']} />
          <YAxis type="category" dataKey="name" width={80} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="textSecondary">
          <strong>Prediction:</strong> {prediction} days
        </Typography>
        <Typography variant="body2" color="textSecondary">
          <strong>{confidenceLevel}% Confidence:</strong> {confidence_interval[0]} - {confidence_interval[1]} days
        </Typography>
        <Typography variant="body2" color="textSecondary">
          <strong>Uncertainty:</strong> ±{uncertainty} days ({((uncertainty / prediction) * 100).toFixed(1)}%)
        </Typography>
        <Typography variant="body2" color="textSecondary">
          <strong>Interval Width:</strong> {confidenceWidth} days
        </Typography>
      </Box>
    </Box>
  );
};

export default UncertaintyChart;