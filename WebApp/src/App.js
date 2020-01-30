import React from 'react';
import logo from './logo.svg';
import './App.css';
import BackVideo from './Components/BackVideo.js'
import Overview from './Components/Overview.js'
import Element3 from './Components/Element3.js'
import { CSSTransition } from 'react-transition-group'

class App extends React.Component {

  constructor(props) {

    super(props);
  
    this.state = {
      el1: true,

      el2: false,

      el3: false,

      el4: false,

      el5: false,

      nuc: true
    };
  }
  

  render(){

    return (

      <body>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet"
          href="https://fonts.googleapis.com/css?family=Roboto+Mono"/>
        

        <CSSTransition
        in={this.state.el1}
        timeout={300}
        unmountOnExit
        onEnter={() => this.setState({el5:false})}
        onExited={() => {setTimeout(() => this.setState({el2:true}), 1000)}}
        classNames="fade"
            >

        <BackVideo vidName={require("./Media/time_lapse.mp4")} endFunc={() => {this.setState({el1:false})}}/>

      </CSSTransition>
        

      <CSSTransition
        in={this.state.el2}
        timeout={300}
        unmountOnExit
        onEnter={() => this.setState({el1:false})}
        onExited={() => {setTimeout(() => this.setState({el3:true}), 1000)}}
        classNames="fade"
      >
        <Overview nuc={this.state.nuc} endFunc={() => {this.setState({el2:false})}}/>

      </CSSTransition>



      <CSSTransition
        in={this.state.el3}
        timeout={300}
        unmountOnExit
        onEnter={() => this.setState({el2:false})}
        onExited={() => {setTimeout(() => this.setState({el4:true}), 1000)}}
        classNames="fade"
      >

        <Element3 endFunc={() => {this.setState({el3:false})}} nuc={this.state.nuc}/>

      </CSSTransition>


      <CSSTransition
        in={this.state.el4}
        timeout={300}
        unmountOnExit
        onEnter={() => this.setState({el3:false})}
        onExited={() => {setTimeout(() => this.setState({el5:true}), 1000)}}
        classNames="fade"
      >

        <BackVideo vidName={require("./Media/8x8_Simulation.mp4")} endFunc={() => {this.setState({el4:false})}}/>

      </CSSTransition>
            <CSSTransition
        in={this.state.el5}
        timeout={300}
        unmountOnExit
        onEnter={() => this.setState({el4:false})}
        onExited={() => {setTimeout(() => this.setState({el1:true}), 1000)}}
        classNames="fade"
      >

        {/* <BackVideo vidName={require("./Media/robot.mp4")} endFunc={() => {this.setState({el5:false})}}/> */}

      </CSSTransition>
        
      </body>
      )
  }
}

export default App;
