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
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight
    };
  }

  componentDidMount() {
    window.addEventListener('resize', this.updateDimensions);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateDimensions);
  }

  updateDimensions = () => {
    this.setState({
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight
    });
  };

  render() {
    const { windowWidth, windowHeight } = this.state;

    return (
      <Router>
        <div className="body">
          <Sidebar pageWrapId={"page-wrap"} outerContainerId={"App"} />

          <div className="main-content">
            <Switch>
              <Route path="/about">
                <About
                  windowWidth={windowWidth}
                  windowHeight={windowHeight}
                />
              </Route>
              <Route path="/growth">
                <div>
                  <PageHeading
                    title="Alphagarden"
                    subtitle="Growth"
                    isPortrait={windowHeight > windowWidth}
                  />
                  <BackVideo
                    vidName={require("./Media/time_lapse.mp4")}
                    endFunc={() => { }}
                    isPortrait={windowHeight > windowWidth}
                  />
                </div>
              </Route>
              <Route path="/simulation">
                <div>
                  <PageHeading
                    title="Alphagarden"
                    subtitle="Simulation"
                    isPortrait={windowHeight > windowWidth}
                  />
                  <img src={Grid} className="SimOverlay" alt="grid overlay"/>
                  <BackVideo
                    vidName={require("./Media/simulation.mp4")}
                    endFunc={() => { }}
                    isPortrait={windowHeight > windowWidth}
                  />
                </div>
              </Route>
              <Route path="/robot">
                <RobotVideoPage
                  windowWidth={windowWidth}
                  windowHeight={windowHeight}
                />
              </Route>
              <Route path="/credits">
                <Credits
                  windowWidth={windowWidth}
                  windowHeight={windowHeight}
                />
              </Route>
              <Route exact path="/">
                <Home
                  windowWidth={windowWidth}
                  windowHeight={windowHeight}
                />
              </Route>
            </Switch>
          </div>

          <Switch>
            <Route path="/analysis">
              <Element3
                endFunc={() => { }} nuc={this.state.nuc}
                windowWidth={windowWidth}
                windowHeight={windowHeight}
              />
            </Route>
          </Switch>
        </div>
      </Router>
    )
  }
}

export default App;
