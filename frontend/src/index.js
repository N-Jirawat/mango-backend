import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { BrowserRouter as Router } from 'react-router-dom';  // ห่อแอปพลิเคชันด้วย Router ที่นี่

ReactDOM.render(
  <Router>
    <App />
  </Router>,
  document.getElementById('root')
);
