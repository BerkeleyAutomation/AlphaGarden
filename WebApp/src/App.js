import React from 'react';
import './App.css';
import BackVideo from './Components/BackVideo.js';
import Element3 from './Components/Element3.js';
import Grid from './Media/zoom_grid.svg';
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";
import Sidebar from './Components/Sidebar';
import Home from './Pages/Home';
import About from './Pages/About';
import RobotVideoPage from './Pages/RobotVideoPage';
import Credits from './Pages/Credits';
import PageHeading from './Pages/PageHeading';

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
        <div className="body">
          <Sidebar pageWrapId={"page-wrap"} outerContainerId={"App"} />

          <div className="main-content">
            <Switch>
              <Route path="/about">
                <About />
              </Route>
              <Route path="/growth">
                <div>
                  <PageHeading title="Alphagarden" subtitle="Growth" />
                  <BackVideo vidName={require("./Media/time_lapse.mp4")} endFunc={() => {}}/>
                </div>
              </Route>
              <Route path="/analysis">
                <Element3 endFunc={() => {}} nuc={this.state.nuc}/>
              </Route>
              <Route path="/simulation">
                <div>
                  <PageHeading title="Alphagarden" subtitle="Simulation" />
                  <img src={Grid} className="SimOverlay" alt="grid overlay"/>
                  <BackVideo vidName={require("./Media/simulation.mp4")} endFunc={() => {}}/>
                </div>
              </Route>
              <Route path="/robot">
                <RobotVideoPage />
              </Route>
              <Route path="/credits">
                <Credits />
              </Route>
              <Route exact path="/">
                <Home />
              </Route>
            </Switch>
          </div>
        </div>
      </Router>
    )
  }
}

export default App;
