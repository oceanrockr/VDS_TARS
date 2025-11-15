import React from 'react';
import { Box } from '@mui/material';

const DashboardLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <Box sx={{ p: 3 }}>{children}</Box>;
};

export default DashboardLayout;
