/**
 * T.A.R.S. Cognitive Orchestration Dashboard
 * Main application component with routing and layout
 */
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Box } from '@mui/material';

import DashboardLayout from './components/Layout/DashboardLayout';
import OverviewPage from './pages/OverviewPage';
import AgentsPage from './pages/AgentsPage';
import SimulationPage from './pages/SimulationPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

// Dark theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4',
    },
    secondary: {
      main: '#ff9800',
    },
    background: {
      default: '#0a1929',
      paper: '#1a2332',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Courier New", monospace',
  },
});

function App() {
  const { connect, disconnect, isConnected } = useWebSocket();

  useEffect(() => {
    // Connect to WebSocket on mount
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <DashboardLayout>
            <Routes>
              <Route path="/" element={<Navigate to="/overview" replace />} />
              <Route path="/overview" element={<OverviewPage />} />
              <Route path="/agents" element={<AgentsPage />} />
              <Route path="/simulation" element={<SimulationPage />} />
              <Route path="/explainability" element={<ExplainabilityPage />} />
            </Routes>
          </DashboardLayout>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
