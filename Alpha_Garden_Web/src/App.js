import React from 'react';
import logo from './logo.svg';
import './App.css';
import BackVideo from './Components/BackVideo.js'
import TextOverlay from './Components/TextOverlay.js'
import Element4 from './Components/Element4.js'


class App extends React.Component {

  constructor(props) {

    super(props);

    this.state = {

      page: <BackVideo 

          vidName={require("./Media/time_lapse.mp4")}

          endFunc={() => { this.setState({

            page: <TextOverlay 

                    endFunc= {() => {this.setState({page:<Element4/>})}}/>
          



          })}}/>
    };
  }
  

  render(){

    return (

      <div>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet"
          href="https://fonts.googleapis.com/css?family=Roboto+Mono"/>
        
        {this.state.page}
        
      </div>
      )
  }
}

export default App;
