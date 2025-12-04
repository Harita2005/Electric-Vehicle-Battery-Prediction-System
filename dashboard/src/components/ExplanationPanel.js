import React from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Chip,
  Tooltip,
  IconButton
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const ExplanationPanel = ({ features }) => {
  const getFeatureDescription = (featureName) => {
    const descriptions = {
      'pack_temp_max_30d': 'Maximum pack temperature in the last 30 days',
      'fast_charge_sessions_30d': 'Number of fast charging sessions in the last 30 days',
      'cumulative_cycles': 'Total number of charge-discharge cycles',
      'vehicle_age_years': 'Age of the vehicle in years',
      'thermal_stress_hours_30d': 'Hours of thermal stress (>40°C) in the last 30 days',
      'soc_std_30d': 'Standard deviation of State of Charge over 30 days',
      'current_mean_7d': 'Average current over the last 7 days',
      'voltage_drop': 'Voltage drop under load (internal resistance indicator)',
      'efficiency_7d': 'Energy efficiency over the last 7 days',
      'deep_cycles': 'Number of deep discharge cycles (>50% DoD)'
    };
    
    return descriptions[featureName] || 'Feature contributing to battery health prediction';
  };

  const getFeatureImpact = (contribution) => {
    if (Math.abs(contribution) > 0.1) return 'high';
    if (Math.abs(contribution) > 0.05) return 'medium';
    return 'low';
  };

  const getImpactColor = (impact) => {
    switch (impact) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .replace(/30d/g, '(30d)')
      .replace(/7d/g, '(7d)');
  };

  const maxContribution = Math.max(...features.map(f => Math.abs(f.contribution)));

  return (
    <Box>
      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
        Top factors influencing the battery health prediction:
      </Typography>
      
      {features.map((feature, index) => {
        const impact = getFeatureImpact(feature.contribution);
        const contributionPercent = (Math.abs(feature.contribution) / maxContribution) * 100;
        const isPositive = feature.contribution > 0;
        
        return (
          <Box key={index} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="body2" sx={{ flexGrow: 1, fontWeight: 500 }}>
                {formatFeatureName(feature.name)}
              </Typography>
              <Chip 
                label={impact.toUpperCase()} 
                color={getImpactColor(impact)}
                size="small"
                sx={{ mr: 1 }}
              />
              <Tooltip title={getFeatureDescription(feature.name)}>
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="body2" color="textSecondary" sx={{ minWidth: 100 }}>
                Value: {typeof feature.value === 'number' ? feature.value.toFixed(2) : feature.value}
              </Typography>
              <Typography 
                variant="body2" 
                color={isPositive ? 'error.main' : 'success.main'}
                sx={{ ml: 2 }}
              >
                {isPositive ? 'Decreases' : 'Increases'} health by {Math.abs(feature.contribution * 100).toFixed(1)}%
              </Typography>
            </Box>
            
            <LinearProgress
              variant="determinate"
              value={contributionPercent}
              color={isPositive ? 'error' : 'success'}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
        );
      })}
      
      <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="body2" color="textSecondary">
          <strong>How to interpret:</strong>
          <br />
          • <span style={{ color: '#d32f2f' }}>Red factors</span> are contributing to faster degradation
          <br />
          • <span style={{ color: '#2e7d32' }}>Green factors</span> are helping maintain battery health
          <br />
          • Higher bars indicate stronger influence on the prediction
        </Typography>
      </Box>
    </Box>
  );
};

export default ExplanationPanel;