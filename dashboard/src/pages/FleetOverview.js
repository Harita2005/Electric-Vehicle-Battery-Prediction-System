import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import BatteryChart from '../components/BatteryChart';
import { fetchFleetData } from '../utils/api';

const FleetOverview = () => {
  const [fleetData, setFleetData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadFleetData();
  }, []);

  const loadFleetData = async () => {
    try {
      setLoading(true);
      const data = await fetchFleetData();
      setFleetData(data);
    } catch (err) {
      setError('Failed to load fleet data');
      // Use mock data for demo
      setFleetData(generateMockFleetData());
    } finally {
      setLoading(false);
    }
  };

  const generateMockFleetData = () => {
    const vehicles = [];
    for (let i = 1; i <= 20; i++) {
      vehicles.push({
        vehicle_id: `EV_${i.toString().padStart(4, '0')}`,
        current_soh: 85 + Math.random() * 10,
        predicted_rul: Math.floor(200 + Math.random() * 800),
        risk_level: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
        last_update: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000),
        mileage: Math.floor(20000 + Math.random() * 80000),
        location: ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'][Math.floor(Math.random() * 5)]
      });
    }
    
    return {
      vehicles,
      summary: {
        total_vehicles: vehicles.length,
        avg_soh: vehicles.reduce((sum, v) => sum + v.current_soh, 0) / vehicles.length,
        high_risk_count: vehicles.filter(v => v.risk_level === 'high').length,
        avg_rul: vehicles.reduce((sum, v) => sum + v.predicted_rul, 0) / vehicles.length
      }
    };
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const handleVehicleClick = (vehicleId) => {
    navigate(`/vehicle/${vehicleId}`);
  };

  if (loading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading fleet data...</Typography>
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
      <Typography variant="h4" gutterBottom>
        Fleet Overview
      </Typography>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Vehicles
              </Typography>
              <Typography variant="h4">
                {fleetData.summary.total_vehicles}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Average SoH
              </Typography>
              <Typography variant="h4">
                {fleetData.summary.avg_soh.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                High Risk Vehicles
              </Typography>
              <Typography variant="h4" color="error">
                {fleetData.summary.high_risk_count}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Avg RUL (days)
              </Typography>
              <Typography variant="h4">
                {Math.round(fleetData.summary.avg_rul)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Fleet Health Chart */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Fleet Health Distribution
              </Typography>
              <BatteryChart 
                data={fleetData.vehicles.map(v => ({
                  vehicle_id: v.vehicle_id,
                  soh: v.current_soh,
                  rul: v.predicted_rul
                }))}
                type="scatter"
              />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Distribution
              </Typography>
              <Box sx={{ mt: 2 }}>
                {['high', 'medium', 'low'].map(risk => {
                  const count = fleetData.vehicles.filter(v => v.risk_level === risk).length;
                  const percentage = (count / fleetData.vehicles.length * 100).toFixed(1);
                  return (
                    <Box key={risk} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                          {risk} Risk
                        </Typography>
                        <Typography variant="body2">
                          {count} ({percentage}%)
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={parseFloat(percentage)} 
                        color={getRiskColor(risk)}
                      />
                    </Box>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Vehicle Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Vehicle Details
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Vehicle ID</TableCell>
                  <TableCell>Current SoH (%)</TableCell>
                  <TableCell>Predicted RUL (days)</TableCell>
                  <TableCell>Risk Level</TableCell>
                  <TableCell>Mileage</TableCell>
                  <TableCell>Location</TableCell>
                  <TableCell>Last Update</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {fleetData.vehicles.map((vehicle) => (
                  <TableRow 
                    key={vehicle.vehicle_id}
                    hover
                    onClick={() => handleVehicleClick(vehicle.vehicle_id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>{vehicle.vehicle_id}</TableCell>
                    <TableCell>{vehicle.current_soh.toFixed(1)}</TableCell>
                    <TableCell>{vehicle.predicted_rul}</TableCell>
                    <TableCell>
                      <Chip 
                        label={vehicle.risk_level.toUpperCase()} 
                        color={getRiskColor(vehicle.risk_level)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{vehicle.mileage.toLocaleString()}</TableCell>
                    <TableCell>{vehicle.location}</TableCell>
                    <TableCell>
                      {vehicle.last_update.toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default FleetOverview;