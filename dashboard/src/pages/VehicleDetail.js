import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  Alert,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import BatteryChart from '../components/BatteryChart';
import UncertaintyChart from '../components/UncertaintyChart';
import ExplanationPanel from '../components/ExplanationPanel';
import { fetchVehicleData, fetchVehiclePrediction } from '../utils/api';

const VehicleDetail = () => {
  const { vehicleId } = useParams();
  const navigate = useNavigate();
  const [vehicleData, setVehicleData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadVehicleData();
  }, [vehicleId]);

  const loadVehicleData = async () => {
    try {
      setLoading(true);
      const [data, pred] = await Promise.all([
        fetchVehicleData(vehicleId),
        fetchVehiclePrediction(vehicleId)
      ]);
      setVehicleData(data);
      setPrediction(pred);
    } catch (err) {
      setError('Failed to load vehicle data');
      // Use mock data for demo
      setVehicleData(generateMockVehicleData());
      setPrediction(generateMockPrediction());
    } finally {
      setLoading(false);
    }
  };

  const generateMockVehicleData = () => {
    const now = new Date();
    const timelineData = [];
    
    // Generate 90 days of data
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      timelineData.push({
        timestamp: date,
        soh: 90 - (i * 0.05) + Math.random() * 2 - 1, // Gradual decline with noise
        pack_voltage: 380 + Math.random() * 20,
        pack_current: -50 + Math.random() * 100,
        pack_temp: 25 + Math.random() * 15,
        soc: 20 + Math.random() * 60
      });
    }

    return {
      vehicle_id: vehicleId,
      timeline: timelineData,
      current_status: {
        soh: timelineData[timelineData.length - 1].soh,
        soc: timelineData[timelineData.length - 1].soc,
        pack_voltage: timelineData[timelineData.length - 1].pack_voltage,
        pack_temp: timelineData[timelineData.length - 1].pack_temp,
        mileage: 45000,
        location: 'San Francisco, CA',
        last_charge: new Date(now.getTime() - 6 * 60 * 60 * 1000)
      },
      alerts: [
        { type: 'warning', message: 'Temperature exceeded 40°C during last charge session' },
        { type: 'info', message: 'Recommended maintenance check in 30 days' }
      ]
    };
  };

  const generateMockPrediction = () => ({
    current_soh: 88.5,
    predicted_rul: 456,
    uncertainty: 45,
    confidence_interval: [411, 501],
    risk_level: 'medium',
    top_features: [
      { name: 'pack_temp_max_30d', value: 42.3, contribution: 0.15 },
      { name: 'fast_charge_sessions_30d', value: 8, contribution: 0.12 },
      { name: 'cumulative_cycles', value: 1250, contribution: 0.10 },
      { name: 'vehicle_age_years', value: 2.3, contribution: 0.08 },
      { name: 'thermal_stress_hours_30d', value: 15, contribution: 0.07 }
    ]
  });

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading vehicle data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/')}
          sx={{ mr: 2 }}
        >
          Back to Fleet
        </Button>
        <Typography variant="h4">
          Vehicle {vehicleId}
        </Typography>
      </Box>

      {/* Current Status Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Current SoH
              </Typography>
              <Typography variant="h4">
                {prediction.current_soh.toFixed(1)}%
              </Typography>
              <Chip 
                label={`${prediction.risk_level.toUpperCase()} RISK`}
                color={getRiskColor(prediction.risk_level)}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Predicted RUL
              </Typography>
              <Typography variant="h4">
                {prediction.predicted_rul} days
              </Typography>
              <Typography variant="body2" color="textSecondary">
                ±{prediction.uncertainty} days
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Current SOC
              </Typography>
              <Typography variant="h4">
                {vehicleData.current_status.soc.toFixed(0)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={vehicleData.current_status.soc} 
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pack Temperature
              </Typography>
              <Typography variant="h4">
                {vehicleData.current_status.pack_temp.toFixed(1)}°C
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last updated: {vehicleData.current_status.last_charge.toLocaleTimeString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Battery Health Timeline
              </Typography>
              <BatteryChart 
                data={vehicleData.timeline}
                type="timeline"
                height={300}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Prediction Uncertainty
              </Typography>
              <UncertaintyChart 
                prediction={prediction.predicted_rul}
                confidence_interval={prediction.confidence_interval}
                uncertainty={prediction.uncertainty}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Telemetry Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Voltage & Current
              </Typography>
              <BatteryChart 
                data={vehicleData.timeline}
                type="voltage_current"
                height={250}
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Temperature & SOC
              </Typography>
              <BatteryChart 
                data={vehicleData.timeline}
                type="temp_soc"
                height={250}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Explanation and Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Prediction Explanation
              </Typography>
              <ExplanationPanel features={prediction.top_features} />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alerts & Recommendations
              </Typography>
              <List>
                {vehicleData.alerts.map((alert, index) => (
                  <ListItem key={index} sx={{ px: 0 }}>
                    <ListItemText
                      primary={
                        <Alert severity={alert.type} sx={{ mb: 1 }}>
                          {alert.message}
                        </Alert>
                      }
                    />
                  </ListItem>
                ))}
              </List>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Vehicle Information
              </Typography>
              <Typography variant="body2">
                Mileage: {vehicleData.current_status.mileage.toLocaleString()} miles
              </Typography>
              <Typography variant="body2">
                Location: {vehicleData.current_status.location}
              </Typography>
              <Typography variant="body2">
                Last Charge: {vehicleData.current_status.last_charge.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VehicleDetail;