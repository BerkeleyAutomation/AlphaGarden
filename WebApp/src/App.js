import React from 'react';
import './App.css';
import BackVideo from './Components/BackVideo.js';
import Element3 from './Components/Element3.js';
import Grid from './Media/zoom_grid.svg';
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";

class App extends React.Component {

  constructor(props) {

    super(props);
  
    this.state = {
      el1: true,
      el2: false,
      el3: false,
      el4: false,
      el5: false,
      el6: false,
      el7: false,
      nuc: true, 
    };
  }

  render() {
    return (
      <Router>
        <div>
          <nav>
            <ul>
              <li>
                <Link to="/">Home</Link>
              </li>
              <li>
                <Link to="/about">About</Link>
              </li>
              <li>
                <Link to="/growth">Growth</Link>
              </li>
              <li>
                <Link to="/analysis">Analysis</Link>
              </li>
              <li>
                <Link to="/simulation">Simulation</Link>
              </li>
              <li>
                <Link to="/robot">Robot</Link>
              </li>
              <li>
                <Link to="/credits">Credits</Link>
              </li>
            </ul>
          </nav>
  
          {/* A <Switch> looks through its children <Route>s and
              renders the first one that matches the current URL. */}
          <Switch>
            <Route path="/about">
            </Route>
            <Route path="/growth">
              <div>
                <BackVideo vidName={require("./Media/time_lapse.mp4")} endFunc={() => {}}/>
              </div>
            </Route>
            <Route path="/analysis">
              <Element3 endFunc={() => {}} nuc={this.state.nuc}/>
            </Route>
            <Route path="/simulation">
              <div>
                <img src={Grid} className="SimOverlay" alt="grid overlay"/>
                <BackVideo vidName={require("./Media/simulation.mp4")} endFunc={() => {}}/>
              </div>
            </Route>
            <Route path="/robot">
              <div>
                <div className="videoContainer">
                  <BackVideo vidName={require("./Media/robot_full.mp4")} endFunc={() => {}}/>
                </div>
              </div> 
            </Route>
            <Route path="/credits">
            </Route>
            <Route path="/">
            </Route>
          </Switch>
        </div>
      </Router>
    )
  }
}

export default App;
