import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Slider,
  Button,
  Alert,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RestoreIcon from '@mui/icons-material/Restore';
import BatteryChart from '../components/BatteryChart';
import { fetchWhatIfAnalysis } from '../utils/api';

const WhatIfAnalysis = () => {
  const [selectedVehicle, setSelectedVehicle] = useState('EV_0001');
  const [scenarios, setScenarios] = useState({
    fast_charging_frequency: 50, // Percentage of current level
    max_temperature: 100,        // Percentage of current level
    charging_depth: 100,         // Percentage of current level
    driving_aggressiveness: 100  // Percentage of current level
  });
  const [baselinePrediction, setBaselinePrediction] = useState(null);
  const [whatIfResults, setWhatIfResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadBaseline();
  }, [selectedVehicle]);

  const loadBaseline = async () => {
    // Mock baseline data
    setBaselinePrediction({
      current_soh: 88.5,
      predicted_rul: 456,
      current_factors: {
        fast_charging_frequency: 8, // sessions per month
        max_temperature: 42.3,      // °C
        charging_depth: 85,         // % average DoD
        driving_aggressiveness: 0.7 // 0-1 scale
      }
    });
  };

  const runWhatIfAnalysis = async () => {
    setLoading(true);
    try {
      // Mock what-if analysis
      const results = generateMockWhatIfResults();
      setWhatIfResults(results);
    } catch (error) {
      console.error('What-if analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateMockWhatIfResults = () => {
    const baseline_rul = baselinePrediction.predicted_rul;
    
    // Calculate impact based on scenario changes
    let rul_change = 0;
    
    // Fast charging impact (reducing fast charging improves RUL)
    const fc_change = (scenarios.fast_charging_frequency - 100) / 100;
    rul_change += fc_change * -50; // Reducing by 50% adds ~25 days
    
    // Temperature impact
    const temp_change = (scenarios.max_temperature - 100) / 100;
    rul_change += temp_change * -30;
    
    // Charging depth impact
    const depth_change = (scenarios.charging_depth - 100) / 100;
    rul_change += depth_change * -20;
    
    // Driving aggressiveness impact
    const drive_change = (scenarios.driving_aggressiveness - 100) / 100;
    rul_change += drive_change * -15;
    
    const new_rul = Math.max(50, baseline_rul + rul_change);
    const soh_change = rul_change * 0.01; // Rough conversion
    const new_soh = Math.min(100, baselinePrediction.current_soh + soh_change);
    
    return {
      new_rul: Math.round(new_rul),
      new_soh: new_soh,
      rul_change: Math.round(rul_change),
      soh_change: soh_change,
      factor_impacts: {
        fast_charging: Math.round(fc_change * -50),
        temperature: Math.round(temp_change * -30),
        charging_depth: Math.round(depth_change * -20),
        driving: Math.round(drive_change * -15)
      },
      timeline_projection: generateTimelineProjection(new_rul, new_soh)
    };
  };

  const generateTimelineProjection = (rul, soh) => {
    const timeline = [];
    const days = Math.min(rul, 365); // Show up to 1 year
    const daily_degradation = (soh - 80) / days; // Degrade to 80% over RUL period
    
    for (let i = 0; i <= days; i += 7) { // Weekly points
      timeline.push({
        timestamp: new Date(Date.now() + i * 24 * 60 * 60 * 1000),
        soh: Math.max(80, soh - (i * daily_degradation)),
        scenario: 'modified'
      });
    }
    
    return timeline;
  };

  const resetScenarios = () => {
    setScenarios({
      fast_charging_frequency: 100,
      max_temperature: 100,
      charging_depth: 100,
      driving_aggressiveness: 100
    });
    setWhatIfResults(null);
  };

  const getImpactColor = (change) => {
    if (change > 10) return 'success';
    if (change > 0) return 'info';
    if (change > -10) return 'warning';
    return 'error';
  };

  const getScenarioDescription = (key, value) => {
    const descriptions = {
      fast_charging_frequency: `${value}% of current fast charging frequency`,
      max_temperature: `${value}% of current maximum temperature exposure`,
      charging_depth: `${value}% of current charging depth`,
      driving_aggressiveness: `${value}% of current driving aggressiveness`
    };
    return descriptions[key];
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        What-If Analysis
      </Typography>
      
      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        Explore how different charging and driving behaviors could impact battery health and remaining useful life.
      </Typography>

      {/* Vehicle Selection */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Select Vehicle</InputLabel>
            <Select
              value={selectedVehicle}
              onChange={(e) => setSelectedVehicle(e.target.value)}
              label="Select Vehicle"
            >
              {Array.from({ length: 10 }, (_, i) => (
                <MenuItem key={i} value={`EV_${(i + 1).toString().padStart(4, '0')}`}>
                  Vehicle EV_{(i + 1).toString().padStart(4, '0')}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Scenario Controls */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Scenario Configuration
              </Typography>
              
              {baselinePrediction && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" color="textSecondary">
                    Current Prediction: {baselinePrediction.predicted_rul} days RUL, {baselinePrediction.current_soh.toFixed(1)}% SoH
                  </Typography>
                </Box>
              )}

              {Object.entries(scenarios).map(([key, value]) => (
                <Box key={key} sx={{ mb: 3 }}>
                  <Typography variant="body2" gutterBottom sx={{ textTransform: 'capitalize' }}>
                    {key.replace(/_/g, ' ')}
                  </Typography>
                  <Slider
                    value={value}
                    onChange={(e, newValue) => setScenarios(prev => ({ ...prev, [key]: newValue }))}
                    min={10}
                    max={150}
                    step={10}
                    marks={[
                      { value: 50, label: '50%' },
                      { value: 100, label: '100%' },
                      { value: 150, label: '150%' }
                    ]}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(val) => `${val}%`}
                  />
                  <Typography variant="caption" color="textSecondary">
                    {getScenarioDescription(key, value)}
                  </Typography>
                </Box>
              ))}

              <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={runWhatIfAnalysis}
                  disabled={loading}
                >
                  Run Analysis
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={resetScenarios}
                >
                  Reset
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              
              {!whatIfResults ? (
                <Alert severity="info">
                  Configure scenarios and click "Run Analysis" to see the impact on battery health.
                </Alert>
              ) : (
                <Box>
                  {/* Summary */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="h4" color={getImpactColor(whatIfResults.rul_change)}>
                          {whatIfResults.new_rul}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          New RUL (days)
                        </Typography>
                        <Chip 
                          label={`${whatIfResults.rul_change > 0 ? '+' : ''}${whatIfResults.rul_change} days`}
                          color={getImpactColor(whatIfResults.rul_change)}
                          size="small"
                        />
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="h4" color={getImpactColor(whatIfResults.soh_change * 10)}>
                          {whatIfResults.new_soh.toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          New SoH
                        </Typography>
                        <Chip 
                          label={`${whatIfResults.soh_change > 0 ? '+' : ''}${whatIfResults.soh_change.toFixed(1)}%`}
                          color={getImpactColor(whatIfResults.soh_change * 10)}
                          size="small"
                        />
                      </Box>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 2 }} />

                  {/* Factor Impacts */}
                  <Typography variant="subtitle2" gutterBottom>
                    Impact Breakdown
                  </Typography>
                  {Object.entries(whatIfResults.factor_impacts).map(([factor, impact]) => (
                    <Box key={factor} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                        {factor.replace(/_/g, ' ')}:
                      </Typography>
                      <Typography 
                        variant="body2" 
                        color={impact > 0 ? 'success.main' : impact < 0 ? 'error.main' : 'textSecondary'}
                      >
                        {impact > 0 ? '+' : ''}{impact} days
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Timeline Projection */}
        {whatIfResults && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Projected Battery Health Timeline
                </Typography>
                <BatteryChart 
                  data={whatIfResults.timeline_projection}
                  type="timeline"
                  height={300}
                />
                <Alert severity="info" sx={{ mt: 2 }}>
                  This projection shows how the modified behavior could affect battery health over time. 
                  Actual results may vary based on other factors not included in this analysis.
                </Alert>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default WhatIfAnalysis;