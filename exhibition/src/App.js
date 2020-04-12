import React from 'react';
import './App.css';
import BackVideo from './Components/BackVideo.js'
import TextOverlay from './Components/TextOverlay.js'
import Element4 from './Components/Element4.js'


class App extends React.Component {

  constructor(props) {
    const nuc = true;

    super(props);
    const endEl1 = () =>{
        activateEl2();
        this.setState({el1:null});
    }

    const endEl2 = () =>{
        activateEl3();
        this.setState({el2:null})
    }

    const endEl3 = () =>{
        activateEl4();
        this.setState({el3:null})
    }

    const activateEl2 = () =>{
      this.setState({el2:<TextOverlay endFunc={endEl2} nuc={nuc}/>})
    }

    const activateEl3 = () => {
      this.setState({el3:<Element4 endFunc = {endEl3} nuc={nuc}/>})
    }

    const activateEl4 = () => {
      this.setState({el4:<BackVideo vidName={require("./Media/simulation.mp4")} />})
    }
  
    this.state = {
      el1: <BackVideo
                  vidName={require("./Media/time_lapse.mp4")}
                  endFunc={endEl1} />,
      el2: null,

      el3: null,

      el4: null
    };


   
  }
  

  render(){

    return (

      <body>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet"
          href="https://fonts.googleapis.com/css?family=Roboto+Mono"/>
        
        {this.state.el1}
        {this.state.el2}
        {this.state.el3}
        {this.state.el4}
        
      </body>
      )
  }
}

export default App;
